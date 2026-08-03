import cv2
import numpy as np

def NormalizeFace(six_point_face, path, rtvecs, camera):
    text = rtvecs[0]['image_info']
    image_points = np.array([
        [float(text[1]), float(text[2])],  # внешний левый (индекс 0)
        [float(text[3]), float(text[4])],  # внутренний левый (индекс 1)
        [float(text[5]), float(text[6])],  # внутренний правый (индекс 2)
        [float(text[7]), float(text[8])],  # внешний правый (индекс 3)
        [float(text[9]), float(text[10])],  # левый рот (индекс 4)
        [float(text[11]), float(text[12])]  # правый рот (индекс 5)
    ], dtype=np.float64)

    # 1. 3D-модель - ТРАНСПОНИРУЕМ!
    model_points_raw = np.array(six_point_face['model'], dtype=np.float64)

    rotation_matrix, _ = cv2.Rodrigues(rtvecs[0]["rotation_vector"])
    model_points = (rotation_matrix.T @ model_points_raw).T

    #model_points = model_points_raw.T  # <-- (3,6) -> (6,3)

    camera_matrix = camera['cameraMatrix'].reshape(3, 3).astype(np.float64)
    dist_coeffs = camera['distCoeffs'].reshape(-1, 1).astype(np.float64)

    projected_points, _ = cv2.projectPoints(
        model_points,
        rtvecs[0]["rotation_vector"],
        rtvecs[0]["translation_vector"],
        camera_matrix,
        dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)

    # 4. Draw the projected points on an empty 640x480 canvas
    canvas = cv2.imread(path+'/'+rtvecs[0]["image_info"][0])

    # Display the image using [PyImageSearch] or [Stack Overflow] techniques
    # cv2.imshow("Projected 3D Points", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    M, _ = cv2.estimateAffinePartial2D(
        image_points, projected_points
    )

    h, w = canvas.shape[:2]
    normalized = cv2.warpAffine(canvas, M, (w, h))

    # Левый глаз (индексы 0 и 1 — внешний и внутренний)
    left_center_x = int((projected_points[0][0] + projected_points[1][0]) / 2)
    left_center_y = int((projected_points[0][1] + projected_points[1][1]) / 2)

    # Правый глаз (индексы 2 и 3 — внешний и внутренний)
    right_center_x = int((projected_points[2][0] + projected_points[3][0]) / 2)
    right_center_y = int((projected_points[2][1] + projected_points[3][1]) / 2)

    # Вырезаем квадрат 60x60 вокруг центра
    eye_size = 30  # половина размера (итого 60x60)
    left_eye = normalized[
               left_center_y - eye_size: left_center_y + eye_size,
               left_center_x - eye_size: left_center_x + eye_size
               ]
    right_eye = normalized[
                right_center_y - eye_size: right_center_y + eye_size,
                right_center_x - eye_size: right_center_x + eye_size
                ]

    for i in range(6):
        cv2.circle(normalized, np.array(projected_points[i], dtype=int), 2, (0, 0, 255), 2)

    cv2.imshow("Projected 3D Points", normalized)
    cv2.imshow("Left eye", left_eye)
    cv2.imshow("Right eye", right_eye)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. Преобразуем в углы Эйлера
    rotation_matrix, _ = cv2.Rodrigues(rtvecs[0]["rotation_vector"])
    euler_angles = cv2.decomposeProjectionMatrix(
        np.hstack((rotation_matrix, rtvecs[0]["translation_vector"]))
    )[6]

    return {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "euler_angles": euler_angles,
    }