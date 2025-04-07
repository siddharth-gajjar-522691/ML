'''
Implementing tjhe face detection python code for machine learning purpose
Importing the dependencies
implmenting HAAR cascade model algorithm for the object detection
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    