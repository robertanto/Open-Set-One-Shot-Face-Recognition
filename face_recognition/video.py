import cv2
import sys
import numpy as np

# File utils

class ReadImage:
    def __init__(self, mode=0, rgb=True):
        '''
            Read img
            mode: int. 0 - COLOR, 1 - GRAYSCALE, 2 - WITH ALPHA CHANNEL
            rgb: bool. Invert RB channels
        '''
        self.mode = mode
        self.rgb = rgb

    def __call__(self, img_path):
        '''
            Read img
            img_path: string. Path to the file
        '''
        m = cv2.IMREAD_COLOR
        if self.mode == 1:
            m = cv2.IMREAD_GRAYSCALE
        elif self.mode == 2:
            m = cv2.IMREAD_UNCHANGED

        img = cv2.imread(img_path, m)

        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

class OpenCVStream:

    def __init__(self, source=0):
        # Open a video file or an image file or a camera stream
        self.cap = cv2.VideoCapture(source)
    
    def get_color_frame(self):
        hasFrame, frame = self.cap.read()
        if not hasFrame:
            raise RuntimeError('No frames available.')
            
        return frame
    
    def stop(self):
        self.cap.close()