import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.cluster import AgglomerativeClustering

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


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def get_centroid(box):
    """Calculate centroid of a bounding box."""

    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def calculate_distance(box1, box2):
    """Calculate the Euclidean distance between the centers of two bounding boxes."""

    centroid1 = get_centroid(box1)
    centroid2 = get_centroid(box2)

    return np.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)


def calculate_iou_matrix(boxes):
    """Calculate the IoU matrix for a list of bounding boxes."""

    iou_matrix = np.zeros((len(boxes), len(boxes)))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou_matrix[i, j] = calculate_iou(boxes[i], boxes[j])
            iou_matrix[j, i] = iou_matrix[i, j]

    return iou_matrix


def perform_clustering(boxes, iou_threshold=0.5):
    """Perform Agglomerative Hierarchical Clustering on bounding boxes."""

    iou_matrix = calculate_iou_matrix(boxes)

    # Convert the IoU matrix to a distance matrix
    distance_matrix = 1 - iou_matrix

    # Perform clustering using the distance matrix
    clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="single",
                                         distance_threshold=1 - iou_threshold)
    labels = clustering.fit_predict(distance_matrix)

    return labels


def calculate_cluster_info(boxes, labels):
    """Calculate the bounding box and centroid for each cluster."""

    unique_labels = set(labels)
    cluster_info = []

    for label in unique_labels:
        cluster_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == label]

        x1 = min(box[0] for box in cluster_boxes)
        y1 = min(box[1] for box in cluster_boxes)
        x2 = max(box[2] for box in cluster_boxes)
        y2 = max(box[3] for box in cluster_boxes)

        centroid = get_centroid((x1, y1, x2, y2))

        cluster_info.append((centroid, (x1, y1, x2, y2)))

    return cluster_info


def cluster_bounding_boxes(boxes):
    labels = perform_clustering(boxes)
    return calculate_cluster_info(boxes, labels)


def gen_mask_img(anns):
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, :] = 0.5
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img


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

    person_boxes = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if class_ids[i] == 0:
            person_boxes.append((x1, y1, x2, y2))

    final_cluster = cluster_bounding_boxes(person_boxes)

    for (center, bbox) in final_cluster:
        c_x, c_y = center
        c_x, c_y = int(c_x), int(c_y)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (c_x, c_y), 5, (0, 255, 0), -1)

    show(img, bgr=True)


























