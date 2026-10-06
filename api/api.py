import cv2
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import scipy.io as sio
import numpy as np

from create_model.models.lenet import lenet
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from create_model.processing.get_rtvecs import get_rtvecs
from create_model.processing.normalize_face import normalize_face

app = FastAPI()

#.\.venv\Scripts\python.exe -m pip install --upgrade "fastapi[standard]"

class InputImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame: np.ndarray

def gaze_to_pixel(gaze, screen_res=(1920, 1080)):
    # X: [-1, 1] → [0, 1920]
    x_px = int((gaze[0] + 1) / 2 * screen_res[0])

    # Y: [0, 1] → [1080, 0] (инвертируем, т.к. Y растёт вниз)
    y_px = int((1 - gaze[1]) * screen_res[1])

    x_px = max(0, min(screen_res[0], x_px))
    y_px = max(0, min(screen_res[1], y_px))

    return screen_res[0]-x_px, screen_res[1]-y_px

@app.get("/")
def entrance():
    return {"message": "hello!"}

@app.post("/gaze_vector")
def find_gaze_vector(request: InputImage):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # models = NeuroNet.ResNet18().to(device)
    model = lenet().to(device)
    # models = NeuroNet.GazeCNN().to(device)
    checkpoint = torch.load("D:/MPIIGaze/gaze_vector_finder.pth", weights_only=True)
    model.load_state_dict(checkpoint)

    model_path = r'face_landmarker.task'
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    rgb_frame = cv2.cvtColor(request['frame'], cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = landmarker.detect(mp_image)

    # Рисуем точки
    if detection_result.face_landmarks:
        face_points = [0]
        img_h, img_w, _ = request['frame'].shape
        for face_landmarks in detection_result.face_landmarks:
            for (i, landmark) in enumerate(face_landmarks):
                if i in [33, 133, 362, 263, 61, 291]:
                    """Left eye outer corner 33
                    Right eye outer corner 263
                    Left eye inner corner 133
                    Right eye inner corner 362
                    Mouth left corner 61
                    Mouth right corner 291"""
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    cv2.circle(request['frame'], (x, y), 2, (0, 255, 0), -1)
                    face_points.append(x)
                    face_points.append(y)

        # Параметры для функций
        six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')

        camera = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\Camera.mat')

        # cv2.imshow('Face Points', frame)
        a = get_rtvecs(face_points, six_point_face, camera)
        b = normalize_face(six_point_face, a, camera,
                           binary_image=request['frame'])

        left_eye = b["left_eye"]
        right_eye = b["right_eye"]
        head_pose = b["euler_angles"]

        result = model(
            torch.tensor(left_eye, dtype=torch.float32).unsqueeze(0).to(device),
            torch.tensor(right_eye, dtype=torch.float32).unsqueeze(0).to(device),
            torch.tensor(head_pose, dtype=torch.float32).unsqueeze(0).to(device)
        )

        gaze = result.squeeze().cpu().detach().numpy()
        return gaze_to_pixel(gaze)
    return None