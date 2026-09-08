import cv2
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import scipy.io as sio
import numpy as np
import GetNormalize
import NeuroNet
import NormalizeFace

def gaze_to_pixel(gaze_vector, screen_res=(1920, 1080)):
    distance = 600
    sensitivity = 0.8

    # 1. Проецируем в миллиметры (с инверсией Y)
    x_mm = gaze_vector[0] * distance * sensitivity
    y_mm = -gaze_vector[1] * distance * sensitivity  # ← ОДНА инверсия

    # 2. Размер экрана
    screen_width_mm = 530
    screen_height_mm = 290

    # 3. Переводим в пиксели (центр экрана — точка отсчёта)
    x_px = (x_mm / screen_width_mm) * screen_res[0] + screen_res[0] / 2
    y_px = (y_mm / screen_height_mm) * screen_res[1] + screen_res[1] / 2

    # 4. НЕ ИНВЕРТИРУЙ Y ЗДЕСЬ! (уже инвертировали выше)
    # y_px = screen_res[1] - y_px   ← ЗАКОММЕНТИРУЙ!

    # 5. Обрезаем по границам
    x_px = max(0, min(screen_res[0], int(x_px)))
    y_px = max(0, min(screen_res[1], int(y_px)))
    print(f"gaze_y: {gaze[1]}")
    return int(x_px), int(y_px)



screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = NeuroNet.GazeCNN().to(device)
checkpoint = torch.load("D:/MPIIGaze/gaze_vector_finder_inference.pth", weights_only=True)
model.load_state_dict(checkpoint)

# Загружаем модель
model_path = r'D:\MPIIGaze\face_landmarker.task'

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)

# Подключение к веб-камере (0 — стандартная камера) и какие-то дефолтные настройки

w, h = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 5)

# Проверка, удалось ли запустить камеру
if not cap.isOpened():
    print("Не удалось открыть камеру")
else:
    while cap.isOpened():
        # Чтение одного кадра
        ret, frame = cap.read()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
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

            #Параметры для функций
            six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')

            # Параметры камеры
            # Приблизительная матрица камеры (для веб-камеры 640x480)
            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))  # без искажений

            camera = {
                'cameraMatrix': camera_matrix,
                'distCoeffs': dist_coeffs,
            }

            #cv2.imshow('Face Points', frame)
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

            # Центр между глазами
            # eye_center = (
            #     int((face_points[3] + face_points[5]) / 2),
            #     int((face_points[4] + face_points[6]) / 2)
            # )

            x, y = gaze_to_pixel(gaze, (1920, 1080))

            print(f"Точка взгляда: ({x}, {y})")

            screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.circle(screen, (x, y), 20, (0, 0, 255), -1)  # красный круг
            cv2.putText(screen, f"Gaze: ({x}, {y})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Показываем
            cv2.imshow("Gaze Tracker", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # обновление каждую миллисекунду
                cv2.destroyAllWindows()
                break

            #cv2.waitKey(1)
            #cv2.destroyAllWindows()
        else:
            print("Лицо не обнаружено")

# Обязательно освобождаем ресурс камеры
cap.release()