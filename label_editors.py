"""List of functions for editting segmentation masks of RepImages.

All functions must be called with the following signature:
    classes, classwise_mask, objectwise_mask = label_editor(
        classes, classwise_mask, objectwise_mask
    )
"""
from collections import OrderedDict

from PIL import Image
import torch
import torchvision

import utils
import _make_dataset as mkdata


### helper function for label editors
def get_idx(classes):
    return {class_name: classes.index(class_name) for class_name in classes}

def objwise_to_classwise(objectwise_mask):
    return torch.stack([torch.any(masks, dim=0) for masks in objectwise_mask.values()])

def rect_mask(xmin, ymin, xmax, ymax, height, width):
    ret = torch.zeros((height, width), dtype=torch.uint8)
    ret[ymin:ymax, xmin:xmax] = 255
    return ret

def masks_to_rect_masks(masks):
    height, width = masks.size(-2), masks.size(-1)
    boxes = torchvision.ops.masks_to_boxes(masks).to(int)
    ret_masks = [rect_mask(*box, height, width) for box in boxes]
    return torch.stack(ret_masks)

def save_binary_image(img, fname):
    img = img.to(torch.uint8)
    img = torchvision.transforms.functional.to_pil_image(img*(255 if img.max() == 1 else 1))
    img.save(fname)

def save_masked_image(img, mask, fname):
    img = torchvision.transforms.functional.to_pil_image(img * mask)
    img.save(fname)

def merge_burr_and_nut(masks_nut_in_burr, masks_burr, separate=False):
    indices_burr_with_nut = []
    n_burr = len(masks_burr)

    if masks_nut_in_burr.nelement():
        assert masks_burr.nelement(), 'a rep_image has nuts_in_burr in it but not a burr'
        bboxes_burr = masks_to_rect_masks(masks_burr)

        for mask_nut_in_burr in masks_nut_in_burr:
            intersections = torch.logical_and(mask_nut_in_burr, bboxes_burr)
            scores = intersections.reshape(n_burr, -1).count_nonzero(dim=1) / mask_nut_in_burr.sum(dtype=float)
            idx_burr_with_nut = scores.argmax().item()
            indices_burr_with_nut.append(idx_burr_with_nut)
            mask_burr = masks_burr[idx_burr_with_nut]
            mask_burr_with_nut = torch.logical_or(mask_burr, mask_nut_in_burr)
            masks_burr[idx_burr_with_nut] = mask_burr_with_nut
        if separate:
            indices_burr = [i for i in range(n_burr) if i not in indices_burr_with_nut]
            return masks_burr[indices_burr], masks_burr[indices_burr_with_nut]
    
    if separate:
        return masks_burr, torch.tensor([]).view(0, masks_burr.size(-2), masks_burr.size(-1))
    return masks_burr

### list of label editors
def chestnut_2classes(classes, classwise_mask, objectwise_mask):
    """A label editor for chestnut detection. It converts
        ['nut_fine', 'nut_empty', 'nut_in_burr', 'burr']
    into
        ['nut', 'burr'].
    """
    idx = get_idx(classes)
    masks_nut_in_burr = objectwise_mask[idx['nut_in_burr']]
    masks_burr = objectwise_mask[idx['burr']]
    masks_burr = merge_burr_and_nut(masks_nut_in_burr, masks_burr)
      
    masks_nut = torch.concat([
        objectwise_mask[idx['nut_fine']],
        objectwise_mask[idx['nut_empty']],
    ])

    classes = ['nut', 'burr']
    objectwise_mask = OrderedDict({0: masks_nut, 1: masks_burr})
    classwise_mask = objwise_to_classwise(objectwise_mask)

    return classes, classwise_mask, objectwise_mask

def chestnut_3classes(classes, classwise_mask, objectwise_mask):
    """A label editor for chestnut detection. It converts
        ['nut_fine', 'nut_empty', 'nut_in_burr', 'burr']
    into
        ['nut_fine', 'nut_empty', 'burr'].
    """
    idx = get_idx(classes)
    masks_nut_in_burr = objectwise_mask[idx['nut_in_burr']]
    masks_burr = objectwise_mask[idx['burr']]
    masks_burr = merge_burr_and_nut(masks_nut_in_burr, masks_burr)

    classes = ['nut_fine', 'nut_empty', 'burr']
    objectwise_mask = OrderedDict({
        0: objectwise_mask[idx['nut_fine']], 
        1: objectwise_mask[idx['nut_empty']],
        2: masks_burr
    })
    classwise_mask = objwise_to_classwise(objectwise_mask)

    return classes, classwise_mask, objectwise_mask
    
def chestnut_4classes(classes, classwise_mask, objectwise_mask):
    """A label editor for chestnut detection. It converts
        ['nut_fine', 'nut_empty', 'nut_in_burr', 'burr']
    into
        ['nut_fine', 'nut_empty', 'burr', 'burr_with_nut'].
    """
    idx = get_idx(classes)
    masks_nut_in_burr = objectwise_mask[idx['nut_in_burr']]
    masks_burr = objectwise_mask[idx['burr']]
    masks_burr, masks_burr_with_nut = merge_burr_and_nut(masks_nut_in_burr, masks_burr, separate=True)
      
    classes = ['nut_fine', 'nut_empty', 'burr', 'burr_with_nut']
    objectwise_mask = OrderedDict({
        0: objectwise_mask[idx['nut_fine']], 
        1: objectwise_mask[idx['nut_empty']],
        2: masks_burr,
        3: masks_burr_with_nut
    })
    classwise_mask = objwise_to_classwise(objectwise_mask)

    return classes, classwise_mask, objectwise_mask
