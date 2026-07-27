import cv2
import numpy as np

def yes(six_point_face, path, camera):
    # Загружаем 3D-модель - six_point_face

    # 1. 3D-модель - ТРАНСПОНИРУЕМ!
    model_points_raw = np.array(six_point_face['model'], dtype=np.float64)
    model_points = model_points_raw.T  # <-- (3,6) -> (6,3)

    # 3. Параметры камеры
    camera_matrix = camera['cameraMatrix'].reshape(3, 3).astype(np.float64)
    dist_coeffs = camera['distCoeffs'].reshape(-1, 1).astype(np.float64)

    # 2. 2D-точки из аннотации
    with open(path, "r", encoding="utf-8") as f:
        texts = [i.split() for i in f.readlines()]

    rts = []
    for text in texts:
        image_points = np.array([
            [float(text[1]), float(text[2])],  # внешний левый (индекс 0)
            [float(text[3]), float(text[4])],  # внутренний левый (индекс 1)
            [float(text[5]), float(text[6])],  # внутренний правый (индекс 2)
            [float(text[7]), float(text[8])],  # внешний правый (индекс 3)
            [float(text[9]), float(text[10])],  # левый рот (индекс 4)
            [float(text[11]), float(text[12])]  # правый рот (индекс 5)
        ], dtype=np.float64)

        # 5. Запускаем solvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )

        # 6. Преобразуем в углы Эйлера
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = cv2.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, translation_vector))
        )[6]

        #print(euler_angles.flatten())

        rts.append({
            'image_info': text,
            'rotation_vector': rotation_vector,
            'translation_vector': translation_vector,
        })

    return rts