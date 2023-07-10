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
    mask_img[mask_img <= 0] = 0
    mask_img = (mask_img*255).astype(np.uint8)

    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10, 50)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(mask_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    show(mask_img, bgr=True)


    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, x + w, y + h))

    for (x1, y1, x2, y2) in rectangles:
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    show(mask_img, bgr=True)







import cv2

img = cv2.imread('images/7.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1,)
canny = cv2.Canny(blur, 10, 50)

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    rectangles.append((x, y, x + w, y + h))

# merging rectangles
merged_rectangles = []
while len(rectangles) > 0:
    current_rect = rectangles[0]
    rectangles = rectangles[1:]
    merged = True

    while merged:
        merged = False
        i = 0

        while i < len(rectangles):
            x1, y1, x2, y2 = current_rect
            x3, y3, x4, y4 = rectangles[i]

            new_rect = (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))
            if is_overlap((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
                merged = True
                current_rect = new_rect
                rectangles.pop(i)
            else:
                i += 1

    merged_rectangles.append(current_rect)













