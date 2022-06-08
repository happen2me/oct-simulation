###### Paths ######

# raw dataset locations
RAW_DATA_PATTERN = "./data/raw/{data}"

# b-scans, layers, fluids extracted
LAYER_PATTERN = "./data/extract/layers/{data}/{dtype}"
BSCAN_PATTERN = "./data/extract/bscans/{data}"
FLUID_PATTERN = "./data/extract/fluids/{data}"

# splited bscans and layers iinto train, val, test
SPLIT_PATTERN = "./data/splits/{data}/{name}"

# generated datasets
DATASET_PATTERN = "./data/datasets/{data}/{name}"

