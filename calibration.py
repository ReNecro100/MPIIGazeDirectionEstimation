import cv2
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import scipy.io as sio
import mediapipe as mp

import GetNormalize
import NeuroNet
import NormalizeFace

# Точки на экране (x, y)
calibration_points = [
    (100, 100),  # левый верх
    (960, 100),  # центр верх
    (1820, 100),  # правый верх
    (100, 540),  # левый центр
    (960, 540),  # центр
    (1820, 540),  # правый центр
    (100, 980),  # левый низ
    (960, 980),  # центр низ
    (1820, 980),  # правый низ
]

#Ещё раз
calibration_points.extend(calibration_points)

gaze_samples = []  # список gaze для каждой точки

w, h = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 5)
model_path = r'D:\MPIIGaze\face_landmarker.task'
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.FaceLandmarker.create_from_options(options)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuroNet.GazeCNN().to(device)
checkpoint = torch.load("D:/MPIIGaze/gaze_vector_finder.pth", weights_only=True)
model.load_state_dict(checkpoint)

for point in calibration_points:
    # Показать точку на экране
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.circle(screen, point, 30, (0, 0, 255), -1)
    cv2.imshow("Calibration", screen)

    # Ждём, пока пользователь смотрит на точку
    print(f"Смотри на точку {point}. Нажми SPACE.")
    cv2.waitKey(0)

    # Собираем gaze (5 кадров для усреднения)
    gazes = []
    for _ in range(5):
        ret, frame = cap.read()

        if not ret:
            break
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        # Рисуем точки
        if detection_result.face_landmarks:
            face_points = [0]
            img_h, img_w, _ = frame.shape
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
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        face_points.append(x)
                        face_points.append(y)

            # Параметры для функций
            six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')

            # Параметры камеры
            # Приблизительная матрица камеры (для веб-камеры 640x480)

            camera = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\Camera.mat')

            # cv2.imshow('Face Points', frame)
            a = GetNormalize.yes(face_points, six_point_face, camera)
            b = NormalizeFace.NormalizeFace(six_point_face, a, camera,
                                            binary_image=frame)

            left_eye = b["left_eye"]
            right_eye = b["right_eye"]
            head_pose = b["euler_angles"]

            result = model(
                torch.tensor(left_eye, dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor(right_eye, dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor(head_pose, dtype=torch.float32).unsqueeze(0).to(device)
            )

            gaze = result.squeeze().cpu().detach().numpy()
            print(gaze)

        gazes.append(gaze)

    avg_gaze = np.mean(gazes, axis=0)
    gaze_samples.append(avg_gaze)
    print(f"Записано: {avg_gaze}")

cv2.destroyAllWindows()

# Сохраняем
np.save("calibration_gaze.npy", np.array(gaze_samples))
np.save("calibration_points.npy", np.array(calibration_points))