import os
import re
import copy
from glob import glob
import pathlib
from pathlib import Path
import random
import shutil
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.io
import imageio
import cv2
from PIL import Image
from idp_utils.data_handling.constants import (INSTRUMENT_LABELS, 
                                               BSCAN_PATTERN,
                                               LAYER_PATTERN,
                                               FLUID_PATTERN)

###### extract layer and bscans from data  ######

def detect_edges(img):
    '''
    Detect edges vertically for AROI colormap. It returns edges in the shape of (width, #edges)
    '''
    shape = img.shape
    num_edge = 4
    edges = np.zeros((shape[1], num_edge), dtype=int)
    for i in range(shape[1]):
        for j in range(1, num_edge+1):
            # get the upper bound of class j
            idx = np.nonzero(img[:, i] == j)[0][0]
            # if fluids on boundary, get the lower bound of class j-1
            if idx - edges[i-1][j-1] > 15:
                idx = np.nonzero(img[:, i] == j-1)[0][-1]
            edges[i][j-1] = idx
    return edges

def detect_fluids(img, intensities):
    '''
    Detect fluids for AROI colormap. It returns edges in the shape of img
    '''
    labels = [5, 6, 7]
    fluids = np.zeros_like(img)
    for idx, label in enumerate(labels):
        coor_tuples = np.where(img == label)
        if coor_tuples[0].size != 0:
            intensity = intensities[idx]
            coors = zip(coor_tuples[0], coor_tuples[1])
            for coor in coors:
                fluids[coor[0]][coor[1]] = intensity
    return fluids

def extract_data(file_pattern, bscan_key, layermap_key, bscan_format, layermap_format, layer_labels, bscan_folder, layer_folder, 
                       fluid_folder=None, fluid_key=None, valid_slice_indices_fn=None, remove_from=None, n_remove=0, overwrite=True):
    '''
    Extract bscan, corresponding layer images and fluid (if exists) from mat files. Extracted bscans, layers and fluids
    will be saved as .jpg images to specified bscan and layer folders with the original file name.
    If remove_from is specified, n_remove layers randomly sampled from remove_from list will be removed.
    Args:
      file_pattern (str): a pattern leads to mat files, which contains bscans and layermaps, i.e. /path/to/mat/patient*.mat
      bscan_key (str): the retrieval key of the bscan in the mat
      layermap_key (str): the retrieval key of the layermap in the mat
      bscan_format (str): a combination of h(height), w(width), s(slice); their location corresponds to the shape of bscan file, i.e. 'hws'
      layermap_format (str): a combination of w, s, l(layer)
      layer_labels (list(int)): the values to assigned to each layer
      bscan_folder (str): the folder to save bscan slice images
      layer_folder (str): the folder to save layer slice images
      fluid_folder (str): the folder to save fluid slice images
      valid_slice_indices_fn (function): a function to find valid slice indices in a mat
      remove_from (list(int)): a list of layers that can be removed
      n_remove (int): number of layers to remove. It should be smaller than the length of remove_from list
      overwrite (bool): overwrite existing files.
    '''
    # Create folders for bscan, layer, and fluid
    if not Path(bscan_folder).exists():
        Path(bscan_folder).mkdir(parents=True, exist_ok=True)
        print(f"Created bscan folder {bscan_folder}")
    if not Path(layer_folder).exists():
        Path(layer_folder).mkdir(parents=True, exist_ok=True)
        print(f"Created layer folder {layer_folder}")
    if fluid_key is not None and fluid_folder is not None and not Path(fluid_folder).exists():
        Path(fluid_folder).mkdir(parents=True, exist_ok=True)
        print(f"Created fluid folder {layer_folder}")
    
    file_glob = glob(file_pattern)
    print(f"{len(file_glob)} files matches pattern: {file_pattern}")

    # Traverse through each file that matches the given file_pattern
    for f in tqdm(file_glob):
        mat = scipy.io.loadmat(f)
        # Load raw bscan, unplotted layermap and fluid from mat file
        bscan = np.asarray(mat[bscan_key])
        layermap = np.asarray(np.nan_to_num(mat[layermap_key]), dtype=int)
        fluid = None
        if fluid_key is not None:
            fluid = np.asarray(mat[fluid_key], dtype='uint8')
            assert fluid.shape == bscan.shape, f"Fluid should share same width, height, layer format & shape with bscans {bscan.shape}, but got {fluid.shape}"
        
        # Load meta infomation of an image
        assert len(bscan.shape) == 3 and len(bscan_format) == 3, f"bscan is expected to be three dim, but get {bscan_format}({len(bscan.shape)})"
        height = bscan.shape[bscan_format.index('h')]
        width = bscan.shape[bscan_format.index('w')]
        n_slice = bscan.shape[bscan_format.index('s')]
        n_layer = layermap.shape[layermap_format.index('l')]
        assert len(layer_labels) >= n_layer, f"layer_labels should have more elements that #layers({n_layer}), but got length {len(layer_labels)}"
        assert 'w' in layermap_format and 's' in layermap_format and 'l' in layermap_format, f"layermap_format is illegal, got {layermap_format}"
        assert layermap.shape[layermap_format.index('w')] == width, f"width of bscan ({width}) is inconsistent with that of layermap ({layermap.shape[layermap_format.index('w')]})"
        assert layermap.shape[layermap_format.index('s')] == n_slice, f"#slice of bscan ({n_slice}) is inconsistent with that of layermap ({layermap.shape[layermap_format.index('s')]})"
        
        # For some images, not every slice contains labeled information. This filters out useless slices.
        if valid_slice_indices_fn is not None:
            valid_slice_indices = valid_slice_indices_fn(mat)
        else:
            valid_slice_indices = range(n_slice)
        
        # Traverse through each slice in a mat
        for s in valid_slice_indices:
            # check existence, this only works when remove_from is None
            save_name = f.split('/')[-1].split('.')[0] + '_' + str(s) + '.jpg'
            if not overwrite and Path(os.path.join(bscan_folder, save_name)).exists() and Path(os.path.join(layer_folder, save_name)).exists():
                continue
            # Extract bscan slice directly
            if bscan_format.index('s') == 0:
                scan_slice = bscan[s, :, :]
            elif bscan_format.index('s') == 1:
                scan_slice = bscan[:, s, :]
            else: # bscan_format.index('s') == 2
                scan_slice = bscan[:, :, s]
                
            # Plot layer slice with stored layer data
            layer_slice = np.zeros_like(scan_slice)
            # sample layers to remove if remove_from is specified
            layers_remove = None
            if remove_from is not None:
                layers_remove = random.sample(remove_from, n_remove)
                layers_remove = sorted(layers_remove)
            for l in range(n_layer):
                # skip removed layers
                if remove_from is not None and l in layers_remove:
                    continue
                # Programmatically build layermap width indices (this replaces many if clause)
                layer_map_width_indice = [0,0,0]
                layer_map_width_indice[layermap_format.index('s')] = s
                layer_map_width_indice[layermap_format.index('l')] = l
                layer_map_width_indice[layermap_format.index('w')] = range(width)
                layer_map_width_indice = tuple(layer_map_width_indice)
                layer_slice[layermap[layer_map_width_indice], range(width)] = layer_labels[l] # substitute with layer-specific constant
            # This restores the zero-th line of each image back to black pixels (it assumes they used to be black pixels)
            # P.S. I forgot why I need this line
            layer_slice[0, :] = 0
            
            # Extract fluid slice directly if fluid_key is set (therefore fluid is not None)
            fluid_slice = None
            if fluid is not None:
                # fluid and bscan share the same format, therefore here I use bscan_format for fluid
                if bscan_format.index('s') == 0:
                    fluid_slice = fluid[s, :, :]
                elif bscan_format.index('s') == 1:
                    fluid_slice = fluid[:, s, :]
                else: # bscan_format.index('s') == 2
                    fluid_slice = fluid[:, :, s]
            
            # Save each slice as a file
            scan_slice_img = Image.fromarray(scan_slice)
            layer_slice_img = Image.fromarray(layer_slice)
            save_name = f.split('/')[-1].split('.')[0] + '_' + str(s) + '.jpg'
            # If the layer is reduced, append delxxx to its save name
            if remove_from is not None and len(layers_remove) > 0:
                save_name = save_name.split('.')[0] + '_del' + ''.join(str(x) for x in layers_remove) + '.jpg'
            scan_slice_img.save(os.path.join(bscan_folder, save_name))
            layer_slice_img.save(os.path.join(layer_folder, save_name))
            if fluid_slice is not None:
                fluid_slice_img = Image.fromarray(fluid_slice)
                fluid_slice_img.save(os.path.join(fluid_folder, save_name))

def extract_data_aroi(raw_data_folder, bscan_folder, layer_folder, fluid_folder, dtype, 
                            fluid_labels, layer_labels, remove_from=None, n_remove=0,
                            save_extension='png'):
    ''' output: bscans, fluids, layers folders with corresponding files
        args:
            raw_data_folder: source raw data folder
            bscan_folder: destination bscans folder
            layer_folder: destination layers folder
            fluid_folder: destination fluids folder
            dtype: str, name of type of layers
            fluid_labels: fluid label intensities as a list
            layer_labels: layer label intensities as a list
            remove_from (list(int)): a list of layers that can be removed
            n_remove (int): number of layers to remove. It should be smaller than the length of remove_from list
    '''    
    data =  "AROI"
    raw_path = f"24 patient/patient*/raw/labeled/*.png"
    mask_path = f"24 patient/patient*/mask/number/*.png"
    mask_trunk = "24 patient/patient{patient_number}/mask/number/patient{patient_number}_raw{slice_number}.png"
    name_pattern = re.compile("patient([0-9]+)_raw([0-9]+)\.png")
    skipped_files = 0
    
    raw_paths = glob(os.path.join(raw_data_folder, raw_path))
    assert len(raw_paths) != 0
    mask_paths = glob(os.path.join(raw_data_folder, mask_path))
    assert len(mask_paths) != 0
    
    raw_files = [i.split('/')[-1] for i in raw_paths]
    mask_files = [i.split('/')[-1] for i in mask_paths]
    
    if not Path(bscan_folder).exists():
        Path(bscan_folder).mkdir(parents=True)
        print(f"Created folder {bscan_folder}")

    if not Path(layer_folder).exists():
        Path(layer_folder).mkdir(parents=True)
        print(f"Created folder {layer_folder}")

    if not Path(fluid_folder).exists():
        Path(fluid_folder).mkdir(parents=True)
        print(f"Created folder {fluid_folder}")
        
    for raw in tqdm(raw_paths):
        assert raw.split('/')[-1] in mask_files, f"raw image {raw} does not correspond to any image in mask files"

        patient_idx, slice_idx = name_pattern.fullmatch(raw.split('/')[-1]).groups(0)
        mask = mask_trunk.format(patient_number=patient_idx, slice_number=slice_idx)
        mask = os.path.join(raw_data_folder, mask)

        try:
            raw_img = imageio.imread(raw) # height, width
        except Exception as e:
            print(f"[{skipped_files}]Error occurred when opening {raw}")
            print(e)
            skipped_files += 1
            continue
        mask_img = None

        try:
            mask_img = imageio.imread(mask)
        except Exception as e:
            print(f"Error occurred when reading mask", mask)
        if mask_img is None:
            print(f"[{skipped_files}]Read mask is none")
            skipped_files += 1
            continue

        try:
            edges = detect_edges(mask_img) # width, #layers
        except Exception as e:
            print(f"Error occurred when detecting edges", mask)
            skipped_files += 1
            continue

        try:
            fluid_slice = detect_fluids(mask_img, fluid_labels)
        except Exception as e:
            print(f"Error occurred when detecting fluids", mask)
            print(e)
            continue
            
        # label map with layers
        layers_remove = None
        if remove_from is not None:
            layers_remove = random.sample(remove_from, n_remove)
            layers_remove = sorted(layers_remove)

        width, n_layer = edges.shape
        layer_slice = np.zeros_like(raw_img)
        for l in range(n_layer):
            if remove_from is not None and l in layers_remove:
                continue
            layer_slice[edges[:, l], range(width)] = layer_labels[l] # substitute with layer-specific constant

        # Save each slice as a file
        save_name = raw.split('/')[-1].split('.')[0] + '.' + save_extension
        
        scan_slice_img = Image.fromarray(raw_img)
        fluid_slice_img = Image.fromarray(fluid_slice)
        layer_slice_img = Image.fromarray(layer_slice)
        
        scan_slice_img.save(os.path.join(bscan_folder, save_name))
        fluid_slice_img.save(os.path.join(fluid_folder, save_name))
        if remove_from is not None and len(layers_remove) > 0:
            save_name = save_name.split('.')[0] + '_del' + ''.join(str(x) for x in layers_remove) + '.jpg'
        layer_slice_img.save(os.path.join(layer_folder, save_name))

    print("Sum of skipped files: ", skipped_files)

def extract_data_op(raw_data_folder, bscan_folder, layer_folder, layer_labels, instrument_labels=None, save_extension='png'):

    raw_layer_labels = [1, 3]
    raw_instrument_labels = [2, 4]

    if not Path(bscan_folder).exists():
        Path(bscan_folder).mkdir(parents=True)
        print(f"Created folder {bscan_folder}")

    if not Path(layer_folder).exists():
        Path(layer_folder).mkdir(parents=True)
        print(f"Created folder {layer_folder}")

    parts = glob(os.path.join(raw_data_folder, "*"))
    for part in parts:
        print(f"{part.split('/')[-1]} Started")
        folders = glob(os.path.join(part, "*"))
        for folder in tqdm(folders):
            folder_name = folder.split('/')[-1].split('.')[0]
            bscans = glob(os.path.join(folder, "[0-9]*.bmp"))
            layers = glob(os.path.join(folder, "segmentation", "[0-9]*.bmp"))
            assert len(bscans) !=0 and len(layers) != 0 and len(bscans) == len(layers)

            for layer in layers:
                layer_name = folder_name + "-" + layer.split('/')[-1].split('.')[0] + '.' + save_extension
                layer_arr = np.asarray(Image.open(layer))
                for i in range(2):
                    layer_arr[layer_arr == raw_layer_labels[i]] = layer_labels[i]
                if instrument_labels:
                    assert len(instrument_labels) >= len(raw_instrument_labels), \
                        f"instrument_labels ({len(instrument_labels)}) is not enough (expect {len(raw_instrument_labels)})"
                    for i in range(2):
                        layer_arr[layer_arr == raw_instrument_labels[i]] = INSTRUMENT_LABELS[i]
                layer_img = Image.fromarray(layer_arr)
                layer_img.save(os.path.join(layer_folder, layer_name))
            
            for bscan in bscans:
                bscan_name = folder_name + "-" + bscan.split('/')[-1].split('.')[0] + '.' + save_extension
                bscan_arr = np.asarray(Image.open(bscan))
                bscan_img = Image.fromarray(bscan_arr)
                bscan_img.save(os.path.join(bscan_folder, bscan_name))

def get_dme_valid_idx(patient, key='manualLayers1'):
    valid_idx = []
    for i in range(patient[key].shape[-1]):
        x = np.max(np.asarray(np.nan_to_num(patient[key][:,:,i])))
        if x > 0:
            valid_idx.append(i)
    return valid_idx

def get_amd_valid_idx(patient, key='layerMaps'):
    '''
    The 11 B-scans per patient were annotated centered at fovea and 5 frames on either side of the fovea
    This function gives the valid B-scans index
    '''
    idx = []
    for i in range(patient[key].shape[0]):
        x = patient[key][i,:,:]
        if np.sum(np.nan_to_num(x)) != 0:
            idx.append(i)
    return idx

###### split Data into train, val and test  ######

def extract_name(f, extension='jpg'):
    ''' extract the original name of f '''
    pattern = r"(.+)_del.+"
    f_original = re.match(pattern, f).group(1) + "." + extension
    return f_original
    
def split_files(file_names, train_ratio, test_ratio):
    ''' split file names into train, val, test and return as a dictionary'''
    random.shuffle(file_names)
    
    num_train = int(len(file_names) * train_ratio)
    num_test = int(len(file_names) * test_ratio)
    
    train_files = file_names[:num_train]
    test_files = file_names[num_train:num_train+num_test]
    val_files = file_names[num_train+num_test:]
    
    splited_files = { 'train': train_files, 'test': test_files, 'val': val_files}
    return splited_files

def generate_label(layer_path, fluid_path):
    ''' combine layer and fluid as label'''
    layer = imageio.imread(layer_path)
    fluid = imageio.imread(fluid_path)
    
    label = np.where(layer > 0, layer, fluid)
    label_img = Image.fromarray(label)
    return label_img

def prepare_files(data, dst_folder, train_ratio, test_ratio,
                  with_fluids=True, dtype="original", merge_original=False,
                 extension='jpg'):
    ''' output: splitted version of files (labels & bscans) with train, val, test
        args:
            data: str, name of data 
            dst_folder: destination folder
            train_ration, test_ration: obvious
            with_fluids: whether the dataset contains fluids
            dtype: str, name of type of layeys (e.g. could be reduced layers), by default "original"
            merge_original: whether to contain original data with original layers for reduced layers
    '''
    formatted_pattern = BSCAN_PATTERN.format(data=data) if dtype == 'original' \
        else LAYER_PATTERN.format(data=data, dtype=dtype)
    files_pattern = os.path.join(formatted_pattern, '*.' + extension)
    files = glob(files_pattern)
    print(f"[INFO] {len(files)} files matches pattern {files_pattern}")
        
    file_names = [f.split('/')[-1] for f in files]
    
    train_ratio, test_ratio = 0.8, 0.1
    splited_files = split_files(file_names, train_ratio, test_ratio)
            
    src_bscan_folder = BSCAN_PATTERN.format(data=data)
    src_layer_folder = LAYER_PATTERN.format(data=data, dtype=dtype)
    if with_fluids:
        src_fluid_folder = FLUID_PATTERN.format(data=data)
    
    for mode in ['train', 'test', 'val']:
        print(f"Preparing {mode} files:")
        
        dst_label_folder = os.path.join(dst_folder, 'labels', mode)
        Path(dst_label_folder).mkdir(parents=True, exist_ok=True)
        print(f"created dst label folder {dst_label_folder}")
        
        dst_bscan_folder = os.path.join(dst_folder, 'bscans', mode)
        Path(dst_bscan_folder).mkdir(parents=True, exist_ok=True)
        print(f"created dst bscan folder {dst_bscan_folder}")
        
        for f in tqdm(splited_files[mode]):
            
            f_original = extract_name(f, extension) if dtype!="original" and dtype!="hetero" else f
            
            #### Paths with name f ####
            src_layer_path = os.path.join(src_layer_folder, f)

            dst_label_path = os.path.join(dst_label_folder, f)

            dst_bscan_path = os.path.join(dst_bscan_folder, f)
                
            #### Paths with the original name of f ####
            src_bscan_path = os.path.join(src_bscan_folder, f_original)
            
            if with_fluids:
                src_fluid_path = os.path.join(src_fluid_folder, f_original)
            
            #### if merge: bscans & labels of f original ####
            if dtype != "original" and merge_original:
                # bscan path of f_original
                src_original_bscan_path = src_bscan_path
                dst_original_bscan_path = os.path.join(dst_bscan_folder, f_original)
                
                # label paths of f_original
                src_original_layer_folder = LAYER_PATTERN.format(data=data, dtype="original")
                src_original_layer_path = os.path.join(src_original_layer_folder, f_original)
                dst_original_label_path = os.path.join(dst_label_folder, f_original)
                
                if not Path(src_original_bscan_path).exists():
                    print(f"[WARN] src original B-scan does not exist for {f_original}")
                    continue
                if not Path(src_original_layer_path).exists():
                    print(f"[WARN] src original layer does not exist for {f_original}")
                    continue
                    
                # copy bscan of f original 
                # (given that if merge, moving twice would result in error, here assume enough space)
                shutil.copy(src_original_bscan_path, dst_original_bscan_path)
                
                # save(with fluids) or move(without fluids) label of f original
                if with_fluids: # label = layer + fluid
                    if not Path(src_fluid_path).exists():
                        print(f"[WARN] src fluid does not exist for {f_original}")
                    label_original_img = generate_label(src_original_layer_path, src_fluid_path)
                    label_original_img.save(dst_original_label_path)     
                else: #layer
                    shutil.copy(src_original_layer_path, dst_original_label_path)
                
                if not Path(dst_original_bscan_path).exists() or not Path(dst_original_label_path).exists():
                    print(f"[WARN] dst original B-scan or label does not exist for {f_original}")
                    

            #### bscans & labels of f ####
            if not Path(src_bscan_path).exists(): 
                print(f"[WARN] src B-scan does not exist for {f}")
                continue
            if not Path(src_layer_path).exists():
                print(f"[WARN] src layer does not exist for {f}")
                continue
            
            # move bscan of f
            shutil.copy(src_bscan_path, dst_bscan_path)

            # save(with fluids) or move(without fluids) label of f 
            if with_fluids: # label = fluid + layer
                if not Path(src_fluid_path).exists():
                    print(f"[WARN] src fluid does not exist for {f}")
                label_img = generate_label(src_layer_path, src_fluid_path)
                label_img.save(dst_label_path)
            else: # layer
                shutil.copy(src_layer_path, dst_label_path)
                
            if not Path(dst_bscan_path).exists() or not Path(dst_label_path).exists():
                print(f"[WARN] dst B-scan or label does not exist for {f}")
                

def first_occurrence_loss(preds, target, num_classes):
    '''
    This loss computes the difference between the distances from top to the first detection of each class.
    It is normalized by the size of image and by the number of classes.
    '''
    def detect_first_occurrence(img, num_classes):
        '''
        Detect edges vertically for AROI colormap. It returns edges in the shape of (width, #num_classes)
        '''
        shape = img.shape
        # It only works for numpy array
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        edges = np.zeros((shape[1], num_classes), dtype=int)
        for i in range(shape[1]):
            for j in range(1, num_classes+1):
                # get the upper bound of class j
                res = np.nonzero(img[:, i] == j)
                indices = res[0]
                # if the element does not exist, log it as -1
                if len(indices) == 0:
                    idx = -1
                else:
                    idx = indices[0]
                edges[i][j-1] = idx
        return edges
    assert preds.shape == target.shape, f"preds and target are expected to be in the same shape. {preds.shape}!={target.shape}"
    occur_preds = detect_first_occurrence(preds, num_classes)
    occur_target = detect_first_occurrence(target, num_classes)
    loss =  np.abs((occur_target - occur_preds).sum()) / np.prod(preds.shape) / num_classes
    return loss