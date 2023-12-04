import torch
from torchvision import transforms
import cv2
from helper import utility
from PIL import Image
from  models.new_models import custom_v7_2 as model

# path = './model_data/custom/v7.2_Adam_[RAF(AutoAug12x5),FER(AutoAug12x5)]_LR_WB(A)_[L2:0.01]/model.pth'
path = './model_data/custom/v7.2_Adam_[FER]_LR_[L2:0.01]_2nd/model.pth'


def generate_input_frame(frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    grayscale_frame = Image.fromarray(frame)  # convert tp PIL image
    frame_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # turn the graph to single color channel
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize
    ])
    return frame_transforms(grayscale_frame)


def camera(width=1920, high=1080):
    device = utility.select_devices(use_cudnn_if_avaliable=False)
    print('Device:', device)
    print("\n** Press 'esc' to quit **\n")

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 20)  # Position of the text
    font_scale = 0.5
    font_color = (0, 0, 255)  # BGR color
    thickness = 1

    default_capture_device = 0
    capture = cv2.VideoCapture(default_capture_device)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, high)
    capture.open(0)  # turn on capture device

    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    m = model.EmotionCNN(num_classes=7, input_channel=1)
    m.load_state_dict(torch.load(path))
    m.to(device)
    m.eval()
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    with torch.no_grad():
        while capture.isOpened():
            s, frame = capture.read()
            if not s:
                print('Fail to capture, quit')
                break
            if cv2.waitKey(1) in [ord('q'), 27]:  # quit capture, key 'esc'
                print('Quit')
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_image = None
            faces = face_detect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                input_frame = generate_input_frame(face_image).unsqueeze(0).to(device)
                output = m(input_frame)
                probability = torch.nn.functional.softmax(output, dim=1)
                max_probability, prediction = torch.max(probability, dim=1)
                to_decimal = ['{:.2f}'.format(float(item.item())) for item in probability[0]]

                mapping = {labels[i]: to_decimal[i] for i in range(7)}
                text = "|".join([f"{key}: {value}" for key, value in mapping.items()])
                lines = text.split("|")
                y = org[1]
                for line in lines:
                    frame = cv2.putText(frame, line, (org[0], y), font,
                                        font_scale, font_color, thickness, cv2.LINE_AA)
                    y += 30

                frame = cv2.putText(frame, "prediction: " + labels[prediction], (org[0], y), font,
                                    font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                break

            if face_image is None:
                frame = cv2.putText(frame, "No face detected", org, font,
                                    font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            cv2.imshow('camera', frame)  # render frame

    capture.release()


if __name__ == '__main__':
    camera(width=1080, high=720)
