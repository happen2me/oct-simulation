"""Unified labels for AROI and OP datasets.
"""
from collections import namedtuple
from PIL import Image
import numpy as np
from . import constants as C

Label = namedtuple("Label", "instrument mirror ilm ipl rpe bm", defaults=(None,)*6)
unified_label = Label(instrument=1, mirror=2, ilm=3, ipl=4, rpe=5, bm=6)
diff_label = Label(ilm=7, rpe=8) # this is useful when we want to set ilm and rpe with different labels in different datasets
aroi_label = Label(ilm=C.ILM, ipl=C.IPL_INL, rpe=C.RPE, bm=C.BM)
# we did not transform the op dataset, so we directly use the original labels
op_label = Label(instrument=2, mirror=4, ilm=1, rpe=3)


def unify_label(label, src_labels, dst_labels, remove_list=None):
    """Transform values in label from src_labels to dst_labels.
    It will return a new label. The original label will be left untouched.
    Args:
        label (numpy array)
        src_labels (namedtuple(Label))
        dst_labels (namedtuple(Label))
        remove_list (list(int)): all labels in this list will be set to 0
    Returns:
        label (numpy array)
    """
    label_copy = label.copy()
    for l in Label._fields:
        s_label = getattr(src_labels, l)
        d_label = getattr(dst_labels, l)
        if s_label is not None and d_label is not None:
            label_copy[label == s_label] = d_label
    if remove_list is not None:
        for rmv in remove_list:
            label_copy[label == rmv] = 0
    return label_copy


def load_as_array(img_path, label_type=None):
    """Load the image from a path to a numpy array
    If label type is set, it will convert and unify the label.
    label_type accepts None or `op` or `aroi`
    """
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if label_type == 'op':
        img_arr = unify_label(img_arr, src_labels=op_label, dst_labels=unified_label)
    elif label_type == 'aroi':
        img_arr = unify_label(img_arr, src_labels=aroi_label, dst_labels=unified_label,
                              remove_list=C.FLUID_LABELS)
    elif label_type is not None:
        raise ValueError(f"Unknown label type {label_type}")
    return img_arr
