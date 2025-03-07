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
        self.anchors = np.array(anchors)

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
            tx, ty, tw, th = output[..., 0], output[..., 1], output[..., 2], output[..., 3]
            box_confidence = 1 / (1 + np.exp(-output[..., 4, np.newaxis]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))

            cx = np.arange(grid_width).reshape(1, -1, 1)
            cy = np.arange(grid_height).reshape(-1, 1, 1)
            
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            grid = np.stack((cx_grid, cy_grid), axis=-1).reshape((grid_height, grid_width, 1, 2))

            box_xy = (1 / (1 + np.exp(-tx)), 1 / (1 + np.exp(-ty)))
            bx = (box_xy[0] + grid[..., 0]) / grid_width
            by = (box_xy[1] + grid[..., 1]) / grid_height
            
            anchors = self.anchors[i].reshape((1, 1, anchor_boxes, 2))
            bw = (np.exp(tw) * anchors[..., 0]) / image_width
            bh = (np.exp(th) * anchors[..., 1]) / image_height

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs