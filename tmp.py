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


def resize_and_pad(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but single color
        pad_color = [pad_color]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img


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
# for i in range(len(image_files)):
#     img = cv2.imread("outliers/" + image_files[i])

img = cv2.imread("outliers/" + image_files[1])
show(img, bgr=True)


sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
device = "cuda"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)

# show_mask_img(masks)
# show_mask_img(masks, img)

mask_img = gen_mask_img(masks)
show(mask_img)



for mask in masks:
    img[mask['segmentation'] == True] = [np.random.randint(256) for _ in range(3)]
    (x1, y1, w, h) = mask["bbox"]
    x2, y2 = x1+w, y1+h
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
show(img, bgr=True)


rectangles = []
for mask in masks:
    (x1, y1, w, h) = mask["bbox"]
    x2, y2 = x1+w, y1+h
    # x1, y1, x2, y2 = x1+2, y1+2, x2-2, y2-2
    rectangles.append((x1, y1, x2, y2))

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

for (x1, y1, x2, y2) in merged_rectangles:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

show(img, bgr=True)






for mask in masks:
    iou = mask["predicted_iou"]
    if iou > 0.9:
        img[mask['segmentation'] == True] = [np.random.randint(256) for _ in range(3)]
        (x1, y1, w, h) = mask["bbox"]
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

show(img, bgr=True)




with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# yolov3로 사람 검출 얼마나 잘 되는지 확인

image_files = get_image_filenames("outliers")
for i in range(len(image_files)):
    img = cv2.imread("outliers/" + image_files[i])

    tar = img
    height, width, channels = tar.shape

    resized_img = resize_and_pad(tar, (416, 416), pad_color=0)

    blob = cv2.dnn.blobFromImage(resized_img, 0.00392, (416, 416), (0, 0, 0), True, crop=True)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                if height >= width:
                    center_x = int((detection[0]-0.5) * width + width/2)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * height)
                    h = int(detection[3] * height)
                else:
                    center_x = int(detection[0] * width)
                    center_y = int((detection[1]-0.5) * height + height/2)
                    w = int(detection[2] * width)
                    h = int(detection[3] * width)
                # Rectangle coordinates
                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                x2 = int(center_x + w / 2)
                y2 = int(center_y + h / 2)
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # if i in indexes and class_ids[i] == 0:
        if class_ids[i] == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    show(img, bgr=True)

















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




