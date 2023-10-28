import torch
from torchvision import transforms
import cv2
import platform
from PIL import Image
from models import VGG as model


def generate_input_frame(frame):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    grayscale_frame = Image.fromarray(grayscale_frame)  # convert tp PIL image
    frame_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # turn the graph to single color channel
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize
    ])
    return frame_transforms(grayscale_frame)


def camera():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print("\n** Press 'esc' to quit **\n")

    default_capture_device = 0
    
    capture = cv2.VideoCapture(default_capture_device)
    capture.open(0)  # turn on capture device

    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    m = model.VGG16(num_classes=7)
    m.load_state_dict(torch.load("./model_data/VGG16/model.pth"))
    m.to(device)
    m.eval()
    with torch.no_grad():
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
            probability = torch.nn.functional.softmax(output, dim=1)
            max_probability, prediction = torch.max(probability, dim=1)
            to_decimal = ['{:.2f}'.format(float(item.item())) for item in probability[0]]
            mapping = {labels[i]: to_decimal[i] for i in range(7)}
            print(f"Prediction: {labels[prediction]}, probability: {mapping}")

    capture.release()


if __name__ == '__main__':
    camera()
