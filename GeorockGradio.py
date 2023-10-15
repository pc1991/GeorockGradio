#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:51:09 2023

@author: christian
"""

import gradio as gr
import cv2
import requests
import os

from ultralytics import YOLO

!yolo task=detect mode=predict model=yolov8n.pt source="/Users/christian/Downloads/sample-location-photos/1002sl.jpg"

!yolo task=detect mode=predict model=yolov8n-seg.pt source="/Users/christian/Downloads/sample-location-photos/1002sl.jpg"

model = YOLO('yolov8n-seg.pt')
model.predict(source='/Users/christian/Downloads/sample-location-photos/1002sl.jpg')

!yolo task=classify mode=predict model=yolov8n-cls.pt source="/Users/christian/Downloads/sample-location-photos/1002sl.jpg"