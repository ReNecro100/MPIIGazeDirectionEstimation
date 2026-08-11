import cv2
import numpy as np

def yes(strink, six_point_face, camera):
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