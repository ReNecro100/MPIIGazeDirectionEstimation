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
    cv2.imshow("Projected 3D Points", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    M, _ = cv2.estimateAffinePartial2D(
        image_points, projected_points
    )

    h, w = canvas.shape[:2]
    normalized = cv2.warpAffine(canvas, M, (w, h))

    for i in range(6):
        cv2.circle(normalized, np.array(projected_points[i], dtype=int), 2, (0, 0, 255), 2)

    cv2.imshow("Projected 3D Points", normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()