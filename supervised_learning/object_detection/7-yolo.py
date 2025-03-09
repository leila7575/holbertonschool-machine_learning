#!/usr/bin/env python3
"""Yolo v3 algorithm for object detection."""

import numpy as np
from tensorflow import keras as K
import os
import cv2


class Yolo:
    """Class defining yolo v3 algorithm for object detection."""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor for Yolo v3 algorithm."""
        self.model = K.models.load_model(model_path)

        f = open(classes_path, 'r')
        self.class_names = f.read().split("\n")[:-1]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = np.array(anchors)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """ Processes the outputs.
        Returns the boundary boxes, box confidence's
        and class probabilities for each output."""
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidence = self.sigmoid(output[..., 4, np.newaxis])
            class_probs = self.sigmoid(output[..., 5:])
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape
            grid_y = np.indices((grid_height, grid_width, anchor_boxes))[0]
            grid_x = np.indices((grid_height, grid_width, anchor_boxes))[1]

            bx = (self.sigmoid(box[..., 0]) + grid_x) / grid_width
            by = (self.sigmoid(box[..., 1]) + grid_y) / grid_height
            bw = (np.exp(box[..., 2]) * self.anchors[i, :, 0]) / \
                self.model.input.shape[1]
            bh = (np.exp(box[..., 3]) * self.anchors[i, :, 1]) / \
                self.model.input.shape[2]

            box[..., 0] = (bx - bw / 2) * image_width
            box[..., 1] = (by - bh / 2) * image_height
            box[..., 2] = (bx + bw / 2) * image_width
            box[..., 3] = (by + bh / 2) * image_height
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters bounding boxes by confidence score, class probabilities"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box = boxes[i]
            confidence = box_confidences[i]
            class_prob = box_class_probs[i]

            scores = confidence * class_prob
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)
            mask = box_score >= self.class_t

            filtered_boxes.append(box.reshape(-1, 4)[mask.flatten()])
            box_classes.append(box_class.flatten()[mask.flatten()])
            box_scores.append(box_score.flatten()[mask.flatten()])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Selects bounding boxes based on non-maximum suppression."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_labels = np.unique(box_classes)

        for label in unique_labels:
            mask = box_classes == label
            class_boxes = filtered_boxes[mask]
            class_scores = box_scores[mask]

            indices = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[indices]
            class_scores = class_scores[indices]

            while len(class_boxes) > 0:
                highest_conf_box = class_boxes[0]
                highest_conf_score = class_scores[0]

                box_predictions.append(highest_conf_box)
                predicted_box_classes.append(label)
                predicted_box_scores.append(highest_conf_score)

                x1 = np.maximum(highest_conf_box[0], class_boxes[1:, 0])
                y1 = np.maximum(highest_conf_box[1], class_boxes[1:, 1])
                x2 = np.minimum(highest_conf_box[2], class_boxes[1:, 2])
                y2 = np.minimum(highest_conf_box[3], class_boxes[1:, 3])

                intersection_x = np.maximum(0, x2 - x1)
                intersection_y = np.maximum(0, y2 - y1)
                intersection_area = intersection_x * intersection_y

                box_w = (class_boxes[:, 2] - class_boxes[:, 0])
                box_h = (class_boxes[:, 3] - class_boxes[:, 1])
                box_area = box_w * box_h

                highest_conf_w = (highest_conf_box[2] - highest_conf_box[0])
                highest_conf_h = (highest_conf_box[3] - highest_conf_box[1])
                highest_conf_area = highest_conf_w * highest_conf_h
                union = box_area[1:] + highest_conf_area - intersection_area

                iou = intersection_area / union

                iou_mask = iou < self.nms_t
                class_boxes = class_boxes[1:][iou_mask]
                class_scores = class_scores[1:][iou_mask]
        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores)
        )

    @staticmethod
    def load_images(folder_path):
        """Loads images from folder.
        Returns list of images and list of images paths."""
        images = []
        images_paths = []
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                images_paths.append(image_path)
        return images, images_paths

    def preprocess_images(self, images):
        """Resizes images with inter-cubic interpolation
        and normalizes pixel values."""
        image_shapes = []
        pimages = []

        input_w, input_h = self.model.input.shape[1:3]

        for image in images:
            image_height, image_width = image.shape[:2]
            image_shapes.append((image_height, image_width))
            resized_images = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            norm_image = resized_images / 255.0
            pimages.append(norm_image)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Displays the image with boundary boxes, class names and box scores
        Saves the image in the directory detection."""
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_class = int(box_classes[i])
            class_label = self.class_names[box_class]
            score = box_scores[i]
            label = f"{class_label} {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA
            )
        cv2.imshow(str(file_name), image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.isdir("detections"):
                os.makedirs("detections")
            path = os.path.join("detections", os.path.basename(file_name))
            cv2.imwrite(path, image)
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Displays all the images of the folder.
        Returns the boxes, box_classes, box scores for each image
        and the list of image_paths."""
        predictions = []

        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        raw_outputs = self.model.predict(pimages)

        for i, image in enumerate(images):
            outputs = [output[i] for output in raw_outputs]
            boxes, box_confidences, box_class_probs = (
                self.process_outputs(outputs, image_shapes[i])
            )
            filtered_boxes, box_classes, box_scores = (
                self.filter_boxes(boxes, box_confidences, box_class_probs)
            )
            box_predictions, predicted_box_classes, predicted_box_scores = (
                self.non_max_suppression(
                    filtered_boxes, box_classes, box_scores
                )
            )

            predictions.append(
                (box_predictions, predicted_box_classes, predicted_box_scores)
            )

            image_name = os.path.basename(image_paths[i])
            self.show_boxes(
                images[i], box_predictions, predicted_box_classes,
                predicted_box_scores, image_name
            )

        return predictions, image_paths
