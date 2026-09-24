import numpy as np
import cv2

def NormalizeFaceInference(six_point_face, path, camera, data):

    # 1. 3D-модель (исходная)
    model_points_raw = np.array(six_point_face['model'], dtype=np.float64)  # (3, 6)
    model_points = model_points_raw.T  # (6, 3) — для удобства

    # 2. Парсим данные
    data = data.split(" ")
    rotation_vector = np.array([float(data[29]), float(data[30]), float(data[31])], dtype=np.float32)
    translation_vector = np.array([float(data[32]), float(data[33]), float(data[34])], dtype=np.float32)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    model_points = (rotation_matrix.T @ model_points_raw).T

    # 3. Параметры камеры
    camera_matrix = camera['cameraMatrix'].reshape(3, 3).astype(np.float64)
    dist_coeffs = camera['distCoeffs'].reshape(-1, 1).astype(np.float64)

    # 4. Проецируем 3D-точки на изображение (с учётом поворота)
    projected_points, _ = cv2.projectPoints(
        model_points.reshape(-1, 1, 3),  # (6, 1, 3)
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2).astype(np.float32)  # (6, 2)

    # 5. Строим идеальные точки (без поворота)
    R, _ = cv2.Rodrigues(rotation_vector)
    model_points_straight = (R.T @ model_points.T).T  # (6, 3) — убираем поворот
    dst_points, _ = cv2.projectPoints(
        model_points_straight.reshape(-1, 1, 3),
        np.zeros((3, 1)),  # без поворота
        translation_vector,
        camera_matrix,
        dist_coeffs
    )
    dst_points = dst_points.reshape(-1, 2).astype(np.float32)  # (6, 2)

    # 6. Загружаем изображение
    canvas = cv2.imread(path)

    # 7. Строим аффинную матрицу
    M, _ = cv2.estimateAffinePartial2D(projected_points, dst_points)

    h, w = canvas.shape[:2]
    normalized = cv2.warpAffine(canvas, M, (w, h))

    # Левый глаз (индексы 0 и 1 — внешний и внутренний)
    left_center_x = int((projected_points[0][0] + projected_points[1][0]) / 2)
    left_center_y = int((projected_points[0][1] + projected_points[1][1]) / 2)

    # Правый глаз (индексы 2 и 3 — внешний и внутренний)
    right_center_x = int((projected_points[2][0] + projected_points[3][0]) / 2)
    right_center_y = int((projected_points[2][1] + projected_points[3][1]) / 2)

    # Вырезаем квадрат 60x60 вокруг центра
    eye_size_x = 30  # половина размера (итого 60x36)
    eye_size_y = 18
    left_eye = normalized[
               left_center_y - eye_size_y: left_center_y + eye_size_y,
               left_center_x - eye_size_x: left_center_x + eye_size_x
               ]
    right_eye = normalized[
                right_center_y - eye_size_y: right_center_y + eye_size_y,
                right_center_x - eye_size_x: right_center_x + eye_size_x
                ]

    left_eye = left_eye.astype(np.float32) / 255.0
    right_eye = right_eye.astype(np.float32) / 255.0

    for i in range(6):
        cv2.circle(normalized, np.array(projected_points[i], dtype=int), 2, (0, 0, 255), 2)

    # cv2.imshow("Projected 3D Points", normalized)
    # cv2.imshow("Left eye", left_eye)
    # cv2.imshow("Right eye", right_eye)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 6. Преобразуем в углы Эйлера
    euler_angles = cv2.decomposeProjectionMatrix(
        np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    )[6]

    # Нужен выход - gaze vector:
    # day16/0151.jpg

    # small_path = path[0]
    # line = linecache.getline(path + '/' + small_path.split('/')[0] + "/annotation.txt",
    #                          int(small_path.split('/')[1][:4]))
    # line = line.split(' ')
    #
    # gaze_vector = np.array([float(line[26]), float(line[27]), float(line[28])])
    return {
        "left_eye": left_eye.transpose(2, 1, 0),
        "right_eye": right_eye.transpose(2, 1, 0),
        "euler_angles": euler_angles,
        # "gaze_vector": gaze_vector,
    }