import matplotlib.pyplot as plt
import torchvision.utils

def show_image(tensor, mask=None):
    if tensor.ndim == 2:
        tensor = tensor[None]
    if mask is not None:
        tensor = apply_mask(tensor, mask)
    ndarr = tensor.numpy().transpose((1, 2, 0))
    if ndarr.shape[-1] == 1:
        ndarr = ndarr[:, :, 0]
    plt.imshow(ndarr)
    plt.axis('off')
    plt.tight_layout()

def apply_mask(image, mask):
    if image.ndim == 3:
        mask = mask[None]
    return image * mask

def make_mask_grid(image, classwise_mask, objectwise_mask, i_class):
    objs = objectwise_mask[i_class]
    image_class_mask = apply_mask(image, classwise_mask[i_class])
    image_object_masks = [apply_mask(image, obj) for obj in objs]
    grid = torchvision.utils.make_grid(
        [image, image_class_mask] + image_object_masks,
        pad_value=0.5
    ) 
    return grid
