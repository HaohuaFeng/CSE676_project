import time

import torch
import torch.nn.functional as F
from torchvision import models
import cv2
import platform


def camera():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    default_capture_device = 1 if platform.system() == 'Darwin' else 0  # default camera: 1:Mac, 0: other
    capture = cv2.VideoCapture(default_capture_device)
    capture.open(0)  # turn on capture device

    print("\n** Press 'esc' to quit **\n")

    while capture.isOpened():
        s, frame = capture.read()  # capture 1 frame
        if not s:
            print('Fail to capture, quit')
            break

        if cv2.waitKey(1) in [ord('q'), 27]:  # quit capture, key 'esc'
            print('Quit')
            break

        cv2.imshow('camera', frame)  # render frame

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        # call model here, use grayscale_frame as image input

    capture.release()


if __name__ == '__main__':
    camera()
