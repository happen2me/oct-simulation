import numpy as np
import torch


def horizontal_expand_rightward(label, feature, to_expand=20):
    """Expand the specified feature rightword by to_expand pixels"""
    if isinstance(label, torch.Tensor):
        label_copy = label.clone()
    else:
        label_copy = label.copy()
    x, y = np.where(label_copy == feature)
    xacc, yacc = x, y
    for i in range(to_expand):
        # To make sure the expansion does not exceed the image boundary
        expansion_x = x # expand rightward, x axis stays the same
        expansion_y = y + i
        # filter out the pixels that exceed the image boundary
        expansion_x = expansion_x[expansion_y < label.shape[1]]
        expansion_y = expansion_y[expansion_y < label.shape[1]]
        xacc = np.concatenate([xacc, expansion_x])
        yacc = np.concatenate([yacc, expansion_y])
    label_copy[xacc, yacc] = feature
    return label_copy


def vertical_expand_upward(label, feature):
    """Fill all space above the appeared feature
    with feature

    label: expected to be of shape (height, width)
    """
    if isinstance(label, torch.Tensor):
        label_copy = label.clone()
    else:
        label_copy = label.copy()
    height, width = label.shape
    for w in range(width):
        feature_h = np.where(label[:, w] == feature)[0]
        if len(feature_h) == 0:
            continue
        max_h = feature_h[-1]
        label_copy[:max_h, w] = feature
    return label_copy


def expand_label(label, instrument_label=2, mirror_label=4, expansion_instrument=30,
                 expansion_mirror=60, expand_upward=False):
    """Expand the line-shaped instrument label and shadow label into greater but
    inaccurate segmentation mask that covers the whole instrument and shadow.

    Args:
    label: input size is expected to be [h, w]
    """
    # For label 2 4 (instrument & its mirroring), we horizontally expand
    # a couple of pixels rightward
    label = horizontal_expand_rightward(label, instrument_label,
                                        to_expand=expansion_instrument)
    # shadows are generally broader
    label = horizontal_expand_rightward(
        label, mirror_label, to_expand=expansion_mirror)
    if expand_upward:
        label = vertical_expand_upward(label, mirror_label)
        label = vertical_expand_upward(label, instrument_label)
    return label


def get_dst_shadow(src_label, dst_label, instrument_label=2, mirror_label=4,
                   top_layer_label=1, margin_above=0, pad_left=0, pad_right=0):
    """Get the shadow of the source label in the destination label, taking
    the instrument and shadow label in the source as well as the layer label
    in the destination into account
    Args:
    overflow_above: the margin to include above the top layer. This is for the cases
        that the human labeled top layer is inaccurate and leaves some pixels out.
    pad_left: the number of pixels to pad to the left of the shadow. This is for the
        that direct under of the instrument actually includes some layers.
    pad_right: the number of pixels to pad to the right of the shadow. This is for the
        that direct under of the instrument actually includes some layers.
    """
    img_height, img_width = src_label.shape
    shadow_x = np.array([], dtype=np.int64)
    shadow_y = np.array([], dtype=np.int64)
    # Requirements for the shadow label:
    # 1. Horizontally after the starting of the instrument/mirroring & before the
    #    ending of the instrument/mirroring
    # 2. Vertically below the upper bound of layers
    x_src_tool, y_src_tool = np.where(np.logical_or(src_label == instrument_label,
                    src_label == mirror_label))  # (1024, 512)
    if len(x_src_tool) == 0:
        return shadow_x, shadow_y
    left_bound = np.min(y_src_tool)
    right_bound = np.max(y_src_tool)
    # Detect left break and right break of the top layer, this is to adjust the left and
    # right bound of the shadow.
    for y in range(left_bound, img_width):
        # If the layer continues to present to the right of left_bound below the tools,
        # increase left_bound
        if np.any(src_label[:, y] == top_layer_label):
            left_bound = y
        else:
            break
    for y in range(right_bound, -1, -1):
        # If the layer continues to present to the left of right_bound below the tools,
        # decrease right_bound
        if np.any(src_label[:, y] == top_layer_label):
            right_bound = y
        else:
            break
    if pad_left + left_bound < right_bound:
        left_bound += pad_left
    if right_bound - pad_left > left_bound:
        right_bound -= pad_right
    accumulated_min_upperbound = 0
    for i in range(left_bound, right_bound):
        top_layer = np.where(dst_label[:, i] == top_layer_label)[0]
        if len(top_layer) == 0:
            if accumulated_min_upperbound == 0:
                continue
            else:
                # set to current recorded highest layer
                top_layer_upperbound = accumulated_min_upperbound
        else:
            # print("instrument_above", instrument_above, len(instrument_above))
            top_layer_upperbound = np.min(top_layer)
            if top_layer_upperbound - margin_above > 0:
                top_layer_upperbound -= margin_above
            if accumulated_min_upperbound == 0:
                # initialize
                accumulated_min_upperbound = top_layer_upperbound
            else:
                accumulated_min_upperbound = min(
                    accumulated_min_upperbound, top_layer_upperbound)
        x_vertical = np.arange(top_layer_upperbound,
                               img_height)  # upperbound to bottom
        y_vertical = np.full_like(x_vertical, i)
        shadow_x = np.concatenate([shadow_x, x_vertical])
        shadow_y = np.concatenate([shadow_y, y_vertical])
    return shadow_x, shadow_y


def get_shadow_below_instruments(label, instrument_label=2, shadow_label=4, img_height=1024):
    """Get the shadow of the source label, without taking the target label into
    account. Every pixel below the instrument and shadows are covered.
    """
    shadow_x = np.array([], dtype=np.int64)
    shadow_y = np.array([], dtype=np.int64)
    # Requirements for the shadow label:
    # 1. Horizontally after the starting of the instrument/mirroring & before the
    #    ending of the instrument/mirroring
    # 2. Vertically below the lower bound of instrument/mirroring
    x, y = np.where(np.logical_or(label==instrument_label, label==shadow_label)) # (1024, 512)
    if len(x) == 0:
        return shadow_x, shadow_y
    left_bound = np.min(y)
    right_bound = np.max(y)
    accumulated_min_lowerbound = 0
    for i in range(left_bound, right_bound):
        instrument_above = np.where(np.logical_or(label[:, i] == instrument_label, label[:, i] == shadow_label))[0]
        if len(instrument_above) == 0:
            if accumulated_min_lowerbound == 0:
                continue
            else:
                # set to current recorded lowest shadow
                instrument_lowerbound = accumulated_min_lowerbound
        else:
            # print("instrument_above", instrument_above, len(instrument_above))
            instrument_lowerbound = np.max(instrument_above)
            if accumulated_min_lowerbound == 0:
                # initialize
                accumulated_min_lowerbound = instrument_lowerbound
            else:
                accumulated_min_lowerbound = max(accumulated_min_lowerbound, instrument_lowerbound)
        x_vertical = np.arange(instrument_lowerbound, img_height) # upperbound to bottom
        y_vertical = np.full_like(x_vertical, i)
        shadow_x = np.concatenate([shadow_x, x_vertical])
        shadow_y = np.concatenate([shadow_y, y_vertical])
    return shadow_x, shadow_y


def get_shadow_below_top_layer(label, instrument_label=2, shadow_label=4, top_layer_label=1,
                               img_width=512, img_height=1024):
    """Covers instruments, mirroring and the shadows below the highest point of
    upper layers
    """
    shadow_x = np.array([], dtype=np.int64)
    shadow_y = np.array([], dtype=np.int64)
    # Requirements for the shadow label:
    # 1. Horizontally after the starting of the instrument/mirroring & before the
    #    ending of the instrument/mirroring
    # 2. Vertically below the (upperbound of) label 1
    x, y = np.where(np.logical_or(label==instrument_label,
                                  label==shadow_label)) # (1024, 512)
    if len(x) == 0:
        return shadow_x, shadow_y
    left_bound = np.min(y)
    right_bound = np.max(y)
    x, y = np.where(label==top_layer_label)
    upper_bound = np.min(x)
    left_end = upper_bound
    right_end = upper_bound
    for i in (left_bound, 0, -1):
        left_1 = np.where(label[:, i]==top_layer_label)[0]
        if len(left_1) > 0:
            left_end = left_1[0]
            break
    for i in range(right_bound, img_width):
        right_1 = np.where(label[:, i]==top_layer_label)[0]
        if len(right_1) > 0:
            right_end = right_1[0]
            break
    upper_bound = max(upper_bound, min(left_end, right_end))
    for i in range(left_bound, right_bound):
        x_vertical = np.arange(upper_bound, img_height) # upperbound to bottom
        y_vertical = np.full_like(x_vertical, i)
        shadow_x = np.concatenate([shadow_x, x_vertical])
        shadow_y = np.concatenate([shadow_y, y_vertical])
    return shadow_x, shadow_y
