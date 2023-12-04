'''
Intergrate emotion recognition model with real time web-cam feed
Crop the frame of area of a face and use this frame as input to model, 
filter out unnecessary disstructed elements in the environment to improve the accuracy.
'''

import torch
from torchvision import transforms
import cv2
from helper import utility
from PIL import Image
from  models.new_models import custom_v7_2 as model


# transform image to grayscale, output as tensor
def generate_input_frame(frame, size=64):
    grayscale_frame = Image.fromarray(frame)
    frame_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # turn the graph to single color channel
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # normalize
    ])
    return frame_transforms(grayscale_frame)


def camera(path, width=1080, high=720):
    device = utility.select_devices(use_cudnn_if_avaliable=False)
    print('Device:', device)
    print("\n** Press 'esc' to quit **\n")

    # set prediction font type and overlay position
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 20)  # origin position of the text
    font_scale = 0.5
    font_color = (0, 0, 255)  # BGR color code
    thickness = 1

    default_capture_device = 0 # device ID
    capture = cv2.VideoCapture(default_capture_device)

    # set frame high and width
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, high)

    # turn on capture device
    capture.open(0)

    # use haarcascades from cv2 for face detection
    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # import model and load trained parameter
    m = model.EmotionCNN(num_classes=7, input_channel=1)
    m.load_state_dict(torch.load(path))
    m.to(device)
    m.eval()

    # names of 7 classes
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    with torch.no_grad():
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                print('Fail to capture, quit')
                break
            if cv2.waitKey(1) in [ord('q'), 27]:  # quit capture, key: 'esc'
                print('Quit')
                break
            
            # turn captured frame to grayscale fro faster computation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            face_image = None
            faces = face_detect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

            # x, y: the origin of the box that contains face
            # w, h: width and high of the box
            for (x, y, w, h) in faces:
                # box the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # crop the frame to get only the area of the box
                face_image = frame[y:y+h, x:x+w]

                # turn the image to model input tensor data
                input_frame = generate_input_frame(face_image).unsqueeze(0).to(device)

                # model output
                output = m(input_frame)

                # use softmax to normalize and turn the output to probabilities of classes
                probability = torch.nn.functional.softmax(output, dim=1)
                # select the class with the highest probability
                prediction = torch.argmax(probability, dim=1)
                # show only two decimal places of the probabilities
                to_decimal = ['{:.2f}'.format(float(item.item())) for item in probability[0]]

                # maps the probabilities to the names of classes
                mapping = {labels[i]: to_decimal[i] for i in range(7)}

                # turn the map into string with k:v seprated by character '|'
                text = "|".join([f"{key}: {value}" for key, value in mapping.items()])
                lines = text.split("|")

                # print probabilities of classes on the captured frame
                y = org[1] # origin y value
                for line in lines:
                    # print the text "class: probability" line by line
                    frame = cv2.putText(frame, line, (org[0], y), font,
                                        font_scale, font_color, thickness, cv2.LINE_AA)
                    # incrementing y axis value for next line
                    y += 30 

                # print prediction after last line of "class: probability"
                frame = cv2.putText(frame, "prediction: " + labels[prediction], (org[0], y), font,
                                    font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                # only capture one face to reduce computation
                break
            
            # pause passing input to model if no face is detected
            if face_image is None:
                frame = cv2.putText(frame, "No face detected", org, font,
                                    font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            # render the frame with face box and text of probabilities of 7 emotions
            cv2.imshow('emotion recognition | Press key "esc" to quit', frame)

    capture.release()


if __name__ == '__main__':
    # path of the trained model parameter, .pth files
    # path = './model_data/custom/v7.2_Adam_[RAF(AutoAug12x5),FER(AutoAug12x5)]_LR_WB(A)_[L2:0.01]/model.pth'
    path = './model_data/custom/v7.2_Adam_[FER]_LR_[L2:0.01]_2nd/model.pth'
    camera(path = path)
