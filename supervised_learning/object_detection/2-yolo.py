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

    def process_outputs(self, outputs, image_size):
        """ Processes the outputs.
        Returns the boundary boxes, box confidence's
        and class probabilities for each output."""
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box_xy, box_wh = output[..., :2], output[..., 2:4]
            box_confidence, class_prob = output[..., 4:5], output[..., 5:]

            box_xy = 1 / (1 + np.exp(-box_xy))
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            class_prob = 1 / (1 + np.exp(-class_prob))

            grid_x, grid_y = np.meshgrid(
                np.arange(grid_width), np.arange(grid_height)
            )
            grid = np.stack(
                (grid_x, grid_y), axis=-1
            ).reshape((grid_height, grid_width, 1, 2))

            box_xy = (box_xy + grid) / [grid_width, grid_height]
            anchors = self.anchors[i].reshape((1, 1, anchor_boxes, 2))
            box_wh = np.exp(box_wh) * anchors
            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)
            box = np.concatenate([box_x1y1, box_x2y2], axis=-1)

            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_prob)

        return boxes, box_confidences, box_class_probs
    
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters bounding boxes based on confidence score and class probabilities."""
        filtered_boxes = []
        box_classes = []
        box_scores = []
        
        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)
            mask = box_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])
            
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)
        
        return filtered_boxes, box_classes, box_scores
