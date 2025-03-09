#!/usr/bin/env python3
"""Yolo v3 algorithm for object detection."""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """Class defining yolo v3 algorithm for object detection."""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor for Yolo v3 algorithm."""
        self.model = K.models.load_model(model_path)

        f = open(classes_path, 'r')
        self.class_names = f.read().split("\n")[:-1]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

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
