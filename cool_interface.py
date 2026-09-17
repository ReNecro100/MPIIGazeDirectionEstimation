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


def gaze_to_pixel(gaze_vector, screen_res=(1920, 1080), distance_to_screen=500):
    """
    gaze_vector: единичный вектор (из модели)
    distance_to_screen: реальное расстояние до экрана в мм (подбери под себя)
    """
    # 1. Масштабируем вектор до реальных миллиметров
    #    z-компонента = расстояние до экрана (известно)
    scale = distance_to_screen / gaze_vector[2]  # масштабный коэффициент

    x_mm = gaze_vector[0] * scale
    y_mm = gaze_vector[1] * scale

    # 2. Размер экрана в мм (подбери под свой монитор)
    screen_width_mm = 530
    screen_height_mm = 290

    # 3. Переводим в пиксели (от центра экрана)
    x_px = (x_mm / screen_width_mm) * screen_res[0] + screen_res[0] / 2
    y_px = (y_mm / screen_height_mm) * screen_res[1] + screen_res[1] / 2

    # 4. Инвертируем Y (OpenCV)
    y_px = screen_res[1] - y_px

    # 5. Обрезаем по границам
    x_px = max(0, min(screen_res[0], int(x_px)))
    y_px = max(0, min(screen_res[1], int(y_px)))

    return int(-x_px+screen_res[0]), int(y_px)


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
landmarker = vision.FaceLandmarker.create_from_options(options)

# Проверка, удалось ли запустить камеру
if not cap.isOpened():
    print("Не удалось открыть камеру")
count = 0
while True:
    # Чтение одного кадра
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

        # focal_length = w
        # center = (w / 2, h / 2)
        # camera_matrix = np.array([
        #     [focal_length, 0, center[0]],
        #     [0, focal_length, center[1]],
        #     [0, 0, 1]
        # ], dtype=np.float32)
        #
        # dist_coeffs = np.zeros((4, 1))  # без искажений
        #
        # camera = {
        #     'cameraMatrix': camera_matrix,
        #     'distCoeffs': dist_coeffs,
        # }

        camera = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\Camera.mat')

        # cv2.imshow('Face Points', frame)
        a = GetNormalize.yes(face_points, six_point_face, camera)
        b = NormalizeFace.NormalizeFace(six_point_face, a, camera,
                                        binary_image=frame)

        left_eye = b["left_eye"]
        right_eye = b["right_eye"]

        head_pose = b["euler_angles"]
        print(f"Real head_pose: {head_pose}")

        result = model(
            torch.tensor(left_eye, dtype=torch.float32).unsqueeze(0).to(device),
            torch.tensor(right_eye, dtype=torch.float32).unsqueeze(0).to(device),
            torch.tensor(head_pose, dtype=torch.float32).unsqueeze(0).to(device)
        )

        gaze = result.squeeze().cpu().detach().numpy()
        #print(f"gaze: {gaze}, z: {gaze[2]}")

        x, y = gaze_to_pixel(gaze, (1920, 1080))

        print(f"Точка взгляда: ({x}, {y})")

        cv2.circle(screen, (x, y), 20, (0, 0, 255), -1)  # красный круг
        cv2.putText(screen, f"Gaze: ({x}, {y})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        count += 1
        cv2.imshow("frame", screen)
        cv2.imshow('left_eye', left_eye.transpose(1,2,0))
        cv2.imshow('right_eye', right_eye.transpose(1, 2, 0))
        if cv2.pollKey() & 0xFF == ord('q'):
            break
    else:
        print("Лицо не обнаружено")

    # Показываем

# Обязательно освобождаем ресурс камеры
landmarker.close()
cap.release()
cv2.destroyAllWindows()