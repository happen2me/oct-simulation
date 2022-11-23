###### Paths ######
from pathlib import Path

ROOT_PATH = Path.cwd().parent
if ROOT_PATH.name != 'IDP':
    print(f"WARNING: root path is at {ROOT_PATH.name}, expect IDP")

# raw dataset locations
RAW_DATA_PATTERN = "data/raw/{data}"

# b-scans, layers, fluids extracted
LAYER_PATTERN = "data/extract/layers/{data}/{dtype}"
BSCAN_PATTERN = "data/extract/bscans/{data}"
FLUID_PATTERN = "data/extract/fluids/{data}"

# splited bscans and layers iinto train, val, test
SPLIT_PATTERN = "data/splits/{data}/"

# generated datasets
DATASET_PATTERN = "data/datasets/{data}"

###### labels: assign value to layers  ######

num_gap = 12 + 1 # self-defined
gap = 255 // num_gap

ILM = 1 * gap # present in 1, 2, 3, 4
RNFL_o = 2 * gap # NFL/FCL in DME, present in 2
IPL_INL = 3 * gap
INL_OPL = 4 * gap
OPL_o = 5 * gap # OPL/ONL in DME
ISM_ISE = 6 * gap
IS_OS = 7 * gap
OS_RPE = 8 * gap

# not sure whether they are the same
RPE = 9 * gap
# RPEDC = 10 * gap
# RPE = 11 * gap

BM = 10 * gap

RTA_LABELS = [ILM, RNFL_o, IPL_INL, INL_OPL, OPL_o, IS_OS, OS_RPE, RPE]
DME_LABELS = [ILM, RNFL_o, IPL_INL, INL_OPL, OPL_o, ISM_ISE, OS_RPE, BM]
AMD_LABELS = [ILM, RPE, BM]  #[19, 171, 190]
AROI_LABELS = [ILM, IPL_INL, RPE, BM] # [19, 57, 171, 190]
HETERO_AROI_LABELS = [ILM+3, IPL_INL, RPE+3, BM] # [22, 57, 174, 190]
OP_LABELS = [ILM, RPE]
FLUID_LABELS = [80, 160, 240]
INSTRUMENT_LABELS = [100, 200] # 100 for real instrument, 200 for reflection

# for segmentation
AROI_LABEL_DICT = { x: i for i, x in enumerate(sorted([0] + AROI_LABELS + FLUID_LABELS))}
