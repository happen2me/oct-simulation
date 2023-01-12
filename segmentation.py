import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import idp_utils.data_handling.constants as C
import argparse


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class ABDataset(Dataset):
    '''
    The dataset where A and B are concatenated horizontally
    '''
    def __init__(self, dataroot, phase='train', grayscale=True, A_transform=None, B_transform=None):
        self.A_transform = A_transform
        self.B_transform = B_transform
        self.grayscale = grayscale
        self.dir_AB = os.path.join(dataroot, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path)
        if self.grayscale:
            AB = AB.convert('L')
        else:
            AB = AB.convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        if self.A_transform:
            A_transform = transforms.Compose(self.A_transform)
            A = A_transform(A)
        if self.B_transform:
            B_transform = transforms.Compose(self.B_transform)
            B = B_transform(B)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}


aroi_label_dict = C.AROI_LABEL_DICT

class Image2Label(object):
    def __init__(self, label_dict):
        self.label_dict = label_dict
    def __call__(self, sample):
        sample = torch.round(sample * 255).type(torch.int64)
        for k, v in self.label_dict.items():
            sample[torch.where(sample==k)] = v
        return sample  


class ModelWrapper(pl.LightningModule):
    def __init__(self, model, lr, loss, num_classes=None):
        super().__init__()
        self.backbone = model
        self.lr = lr
        self.loss = loss
        self.num_classes = num_classes
        self.writer = SummaryWriter()

    def forward(self, x):
        y = self.backbone(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def compute_metrics(self, output, target):
        # 1 is the channel dim
        output = output.argmax(dim=1)
        # restore channel dim for loss computation
        output = output.unsqueeze(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, 'multiclass', ignore_index=-1, num_classes=self.num_classes)
        # then compute metrics with required reduction (see metric docs)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        return {
            'iou': iou_score,
            'f1': f1_score,
            'f2': f2_score,
            'accuracy': accuracy,
            'recall': recall
        }

    def training_step(self, train_batch, batch_idx):
        # Perform the forward pass, compute the loss and the metric of each step
        A = train_batch['A']
        B = train_batch['B']
        A_fake = self.forward(B)
        loss = self.loss(A_fake, A)
        metric_dict = self.compute_metrics(A_fake, A)
        metric = metric_dict['f1']
        return {"loss": loss, "metric": metric}

    def training_epoch_end(self, output):
        loss = 0
        metric = 0
        for o in output:
            # compute the loss and metric of the epoch 
            loss = loss + o['loss']
            metric = metric + o['metric']
        loss = loss / len(output)
        metric = metric / len(output)
        self.writer.add_scalar('Epoch_loss/training', loss, self.current_epoch)
        self.writer.add_scalar('Epoch_metric/training', metric, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        A = val_batch['A']
        B = val_batch['B']
        with torch.no_grad():
            A_fake = self.forward(B)
        loss = self.loss(A_fake, A)
        metric_dict = self.compute_metrics(A_fake, A)
        metric = metric_dict['f1']
        return {"loss": loss, "metric": metric}

    def validation_epoch_end(self,output):
        loss = 0
        metric = 0
        for o in output:
            # Compute the loss and metric of the epoch 
            loss = loss + o['loss']
            metric = metric + o['metric']

        loss = loss / len(output)
        metric = metric / len(output)
        self.log('val_dice', metric)
        self.writer.add_scalar('Epoch_loss/validation', loss, self.current_epoch)
        self.writer.add_scalar('Epoch_metric/validation', metric, self.current_epoch)


def main(args):

    # A is label map, B is b-scan. In segmentation, A will be the target. 
    train_dataset = ABDataset(dataroot='data/datasets/AROI/original',
                        phase='train',
                        A_transform=[transforms.ToTensor(), Image2Label(aroi_label_dict)],
                        B_transform=[transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    val_dataset = ABDataset(dataroot='data/datasets/AROI/original',
                        phase='val',
                        A_transform=[transforms.ToTensor(), Image2Label(aroi_label_dict)],
                        B_transform=[transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # unet = smp.Unet(
    #     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=8,                      # model output channels (number of classes in your dataset)
    #     encoder_depth=3,                # Amount of down- and upsampling of the Unet
    #     decoder_channels=(64, 32,16),   # Amount of channels
    #     encoder_weights = "imagenet",         # Model does not download pretrained weights
    #     activation = 'sigmoid'            # Activation function to apply after final convolution       
    # )

    if args.model_type == 'psp':
        net = smp.PSPNet(
            encoder_name = 'resnet34', 
            encoder_weights = 'imagenet',
            encoder_depth=3,
            in_channels=1,
            classes=8
        )
    else:
        net = smp.Unet(
            in_channels = 1,
            classes=8,
            encoder_name='resnet34', 
            encoder_depth=5, 
            encoder_weights='imagenet',
            decoder_channels=(256, 128, 64, 32, 16),
            activation = 'sigmoid'              
        )


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=28, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=28, shuffle=True)



    model = ModelWrapper(net, args.lr, smp.losses.DiceLoss('multiclass', classes=8, ignore_index=0), num_classes=8)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_dice',
        save_top_k=1,
        mode='max',
        every_n_epochs=1,
        save_last=True
    )
    trainer = pl.Trainer(gpus='1', 
                     auto_select_gpus=True,
                    # precision='bf16', 
                    callbacks=checkpoint_callback,
                    check_val_every_n_epoch=1,
                    log_every_n_steps=5,
                    max_epochs=args.epochs,
                    default_root_dir=f"output/checkpoints/{args.savename}")
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(f'output/{args.savename}.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Unet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--savename', type=str, default='unet',
                        help='save name of model and checkpoint folder')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_type', type=str, default='psp',
                        help='choose from unet, psp, pan, link')

    args = parser.parse_args()
    main(args)