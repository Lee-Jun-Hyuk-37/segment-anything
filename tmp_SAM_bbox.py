import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import matplotlib
matplotlib.use("Qt5Agg")


def show(img, bgr=False):
    plt.figure()
    if bgr:
        plt.imshow(img[:, :, ::-1])
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_image_filenames(directory):
    image_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_filenames.append(filename)
    return image_filenames


def show_mask_img(anns, with_img=None):
    plt.figure()
    if np.any(with_img):
        plt.imshow(with_img)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, :] = 0.5
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def gen_mask_img(anns):
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, :] = 0.5
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img


sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
device = "cuda"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


image_files = get_image_filenames("outliers")
for i in range(len(image_files)):
    img = cv2.imread("outliers/" + image_files[i])

    masks = mask_generator.generate(img)
    mask_img = gen_mask_img(masks)

    mask_img = mask_img[:, :, :3]
    mask_img[mask_img < 0] = 0
    mask_img = (mask_img*255).astype(np.uint8)

    for mask in masks:
        (x, y, w, h) = list(map(int, mask["bbox"]))
        cv2.rectangle(mask_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    show(mask_img)








