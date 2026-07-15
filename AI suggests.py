import cv2
import numpy as np
import scipy.io as sio

# Загружаем 3D-модель
six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')

# 1. 3D-модель - ТРАНСПОНИРУЕМ!
model_points_raw = np.array(six_point_face['model'], dtype=np.float64)
model_points = model_points_raw.T  # <-- (3,6) -> (6,3)

# 2. 2D-точки из аннотации
path = r"D:\MPIIGaze\MPIIGaze\Annotation Subset\p00.txt"
with open(path, "r", encoding="utf-8") as f:
    text = f.readline().split()

# ВНИМАНИЕ: проверьте порядок точек!
# В model_points после транспонирования:
# [0] - левый глаз (внешний)   [4.509, -0.483, 2.397]
# [1] - правый глаз (внешний)  [-2.131, 0.483, -2.397]
# [2] - левый глаз (внутренний) [2.131, 0.483, -2.397]
# [3] - правый глаз (внутренний) [4.509, -0.483, 2.397]
# [4] - левый уголок рта         [-2.629, 6.859, -9.86e-32]
# [5] - правый уголок рта        [2.629, 6.859, -9.86e-32]

image_points = np.array([
    [float(text[1]), float(text[2])],   # Левый глаз (внешний)
    [float(text[7]), float(text[8])],   # Правый глаз (внешний)
    [float(text[3]), float(text[4])],   # Левый глаз (внутренний)
    [float(text[5]), float(text[6])],   # Правый глаз (внутренний)
    [float(text[9]), float(text[10])],  # Левый уголок рта
    [float(text[11]), float(text[12])]  # Правый уголок рта
], dtype=np.float64)

# 3. Параметры камеры
camera = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\Camera.mat')
camera_matrix = camera['cameraMatrix'].reshape(3, 3).astype(np.float64)
dist_coeffs = camera['distCoeffs'].reshape(-1, 1).astype(np.float64)

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

print(f"Поза головы (Pitch, Yaw, Roll): {euler_angles.flatten()}")

# 7. Проверка проекции
projected_points, _ = cv2.projectPoints(
    model_points,
    rotation_vector,
    translation_vector,
    camera_matrix,
    dist_coeffs
)