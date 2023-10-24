import torch
from torchvision import transforms
import cv2
import platform
from PIL import Image
from models import model_1 as model


def generate_input_frame(frame):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    grayscale_frame = Image.fromarray(grayscale_frame)  # convert tp PIL image
    frame_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # turn the graph to single color channel
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize
    ])
    return frame_transforms(grayscale_frame)


def camera():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print("\n** Press 'esc' to quit **\n")

    default_capture_device = 1 if platform.system() == 'Darwin' else 0  # default camera: 1:Mac, 0: other
    capture = cv2.VideoCapture(default_capture_device)
    capture.open(0)  # turn on capture device

    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    m = model.EmotionCNN(num_classes=7)
    m.to(device)

    while capture.isOpened():
        s, frame = capture.read()  # capture 1 frame
        if not s:
            print('Fail to capture, quit')
            break
        if cv2.waitKey(1) in [ord('q'), 27]:  # quit capture, key 'esc'
            print('Quit')
            break

        input_frame = generate_input_frame(frame).unsqueeze(0).to(device)

        cv2.imshow('camera', frame)  # render frame

        output = m(input_frame)
        predicted = output.argmax(dim=1)
        print("Prediction:", labels[predicted])

    capture.release()


if __name__ == '__main__':
    camera()
