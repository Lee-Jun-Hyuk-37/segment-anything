import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

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


def is_overlap(rect1_bottom_left, rect1_top_right, rect2_bottom_left, rect2_top_right):
    # rect1 is to the right of rect2
    if rect1_bottom_left[0] > rect2_top_right[0]:
        return False
    # rect1 is to the left of rect2
    if rect1_top_right[0] < rect2_bottom_left[0]:
        return False
    # rect1 is above rect2
    if rect1_bottom_left[1] > rect2_top_right[1]:
        return False
    # rect1 is below rect2
    if rect1_top_right[1] < rect2_bottom_left[1]:
        return False
    return True




# 그냥
image_files = get_image_filenames("outliers")
for i in range(len(image_files)):
    img = cv2.imread("outliers/" + image_files[i])

img = cv2.imread("outliers/" + image_files[0])






from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
device = "cuda"
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(img)
masks, _, _ = predictor.predict()




# anns

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


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


image_files = get_image_filenames("outliers")
for i in range(len(image_files)):
    img = cv2.imread("outliers/" + image_files[i])

img = cv2.imread("outliers/" + image_files[0])
show(img, bgr=True)


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
device = "cuda"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)

show_anns(masks)

plt.show()

show_mask_img(masks)
show_mask_img(masks, img)

mask_img = gen_mask_img(masks)
show(mask_img)


tar_img = (img + (mask_img[:, :, :3]*255).astype(np.uint8))//2


# gray = cv2.cvtColor((mask_img[:, :, :3]*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 1,)
canny = cv2.Canny(gray, 10, 50)

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    rectangles.append((x, y, x + w, y + h))

#
# # 직사각형 테두리 그리기
# for rect in rectangles:
#     x1, y1, x2, y2 = rect
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# show(img, bgr=True)


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


# 직사각형 테두리 그리기
for rect in merged_rectangles:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

show(img, bgr=True)














gray = cv2.cvtColor((mask_img[:, :, :3]*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=500, minLineLength=100, maxLineGap=100)

for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 == x2 or y1 == y2:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

show(img, bgr=True)




