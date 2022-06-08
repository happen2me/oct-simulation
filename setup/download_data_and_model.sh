###### create directories  ######
mkdir -p data/extract/
mkdir -p data/datasets/
mkdir -p data/raw/
mkdir -p data/splits/

###### download & setup all data ######
#RTA
wget --show-progress -O RTA.zip https://doi.org/10.1371/journal.pone.0133908.s002
unzip RTA.zip -d data/raw
mv data/raw/FiletoShareforPONE data/raw/RTA
rm RTA.zip

#AROI
GDRIVE_ID = 1OI8fcfO3Ams47WW7o3OOJaLIunDIWpoB
if gdown $GDRIVE_ID;
then 
    AROI_PID=$!
    wait $AROI_PID
    unrar x AROI.rar data/raw
    mv data/raw/'AROI - online' data/raw/AROI
    rm AROI.rar
fi
    
#DME
kaggle datasets download -d paultimothymooney/chiu-2015 #DME
unzip chiu-2015.zip -d data/raw 
mv data/raw/2015_BOE_Chiu data/raw/DME
rm chiu-2015.zip
rm -rf data/raw/2015_boe_chiu

# kaggle datasets download -d paultimothymooney/farsiu-2014 #AMD approx 34G

###### download the model ######
# download the pix2pix repo
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
# # download the pretrained model
# chmod -R +x pytorch-CycleGAN-and-pix2pix
# ./pytorch-CycleGAN-and-pix2pix/scripts/download_pix2pix_model.sh edges2shoes

