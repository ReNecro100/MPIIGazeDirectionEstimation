import cv2
import numpy as np
import linecache

def NormalizeFace(six_point_face, rtvecs, camera, path="", binary_image=0):
    text = rtvecs['image_info']
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

    rotation_matrix, _ = cv2.Rodrigues(rtvecs["rotation_vector"])
    model_points = (rotation_matrix.T @ model_points_raw).T

    #model_points = model_points_raw.T  # <-- (3,6) -> (6,3)

    camera_matrix = camera['cameraMatrix'].reshape(3, 3).astype(np.float64)
    dist_coeffs = camera['distCoeffs'].reshape(-1, 1).astype(np.float64)

    projected_points, _ = cv2.projectPoints(
        model_points,
        rtvecs["rotation_vector"],
        rtvecs["translation_vector"],
        camera_matrix,
        dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)

    # 4. Draw the projected points on an empty 640x480 canvas
    if path!="":
        canvas = cv2.imread(path + '/' + rtvecs["image_info"][0])
    else:
        canvas = binary_image

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
        np.hstack((rotation_matrix, rtvecs["translation_vector"]))
    )[6]

    #Нужен выход - gaze vector:
    #day16/0151.jpg


    if path != "":
        small_path = rtvecs["image_info"][0]
        line = linecache.getline(path+'/'+small_path.split('/')[0]+"/annotation.txt", int(small_path.split('/')[1][:4]))
        line = line.split(' ')

        gaze_vector = np.array([float(line[26]),float(line[27]),float(line[28])])


        toreturn = {
            "left_eye": left_eye.transpose(2, 1, 0),
            "right_eye": right_eye.transpose(2, 1, 0),
            "euler_angles": euler_angles,
            "gaze_vector": gaze_vector,
        }
    else:
        cv2.imshow("asdasd", normalized)  # показываем как есть
        cv2.imshow("Left Eye (HWC)", left_eye)  # показываем как есть
        cv2.imshow("Right Eye (HWC)", right_eye)
        cv2.waitKey(0)
        toreturn = {
            "left_eye": left_eye.transpose(2, 1, 0),
            "right_eye": right_eye.transpose(2, 1, 0),
            "euler_angles": euler_angles,
        }

    return toreturn