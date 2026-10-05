import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_rtvecs(strink, six_point_face, camera):
    # Загружаем 3D-модель - six_point_face

    # 1. 3D-модель - ТРАНСПОНИРУЕМ!
    model_points_raw = np.array(six_point_face['model'], dtype=np.float64)
    model_points = model_points_raw.T  # <-- (3,6) -> (6,3)

    # 3. Параметры камеры
    camera_matrix = camera['cameraMatrix'].reshape(3, 3).astype(np.float64)
    dist_coeffs = camera['distCoeffs'].reshape(-1, 1).astype(np.float64)

    # 2. 2D-точки из аннотации


    image_points = np.array([
        [float(strink[1]), float(strink[2])],  # внешний левый (индекс 0)
        [float(strink[3]), float(strink[4])],  # внутренний левый (индекс 1)
        [float(strink[5]), float(strink[6])],  # внутренний правый (индекс 2)
        [float(strink[7]), float(strink[8])],  # внешний правый (индекс 3)
        [float(strink[9]), float(strink[10])],  # левый рот (индекс 4)
        [float(strink[11]), float(strink[12])]  # правый рот (индекс 5)
    ], dtype=np.float64)

    # 5. Запускаем solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    return {
            'image_info': strink,
            'rotation_vector': rotation_vector,
            'translation_vector': translation_vector,
        }

def give_this_function_a_proper_name_someday(image, six_point_face, camera):
    # Загружаем 3D-модель - six_point_face

    # 1. 3D-модель - ТРАНСПОНИРУЕМ!
    model_points_raw = np.array(six_point_face['model'], dtype=np.float64)
    model_points = model_points_raw.T  # <-- (3,6) -> (6,3)

    # 3. Параметры камеры
    camera_matrix = camera['cameraMatrix'].reshape(3, 3).astype(np.float64)
    dist_coeffs = camera['distCoeffs'].reshape(-1, 1).astype(np.float64)

    # 2. 2D-точки из изображения
    # Загружаем модель
    model_path = r'D:\MPIIGaze\face_landmarker.task'

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    image = cv2.imread(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    detection_result = landmarker.detect(mp_image)

    face_points = []

    # Рисуем точки
    if detection_result.face_landmarks:
        img_h, img_w, _ = image.shape
        needed = [33, 133, 362, 263, 61, 291]
        landmarks_dict = {}
        for face_landmarks in detection_result.face_landmarks:
            for (i, landmark) in enumerate(face_landmarks):
                if i in needed:
                    """Left eye outer corner 33
                    Right eye outer corner 263
                    Left eye inner corner 133
                    Right eye inner corner 362
                    Mouth left corner 61
                    Mouth right corner 291"""
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    landmarks_dict[i] = (x, y)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                    #cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            face_points = []
            for idx in needed:
                x, y = landmarks_dict[idx]
                face_points.append(x)
                face_points.append(y)
            break

    face_points = np.array(face_points, dtype=np.float64).reshape(-1, 2)

    # 5. Запускаем solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        face_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    return {
        'image_info': image,
        'rotation_vector': rotation_vector,
        'translation_vector': translation_vector,
    }