from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# Загружаем данные
gaze_samples = np.load("calibration_gaze.npy")    # (9, 3)
points = np.load("calibration_points.npy")         # (9, 2)

# Полиномиальная регрессия 2-й степени
model_x = make_pipeline(PolynomialFeatures(2), LinearRegression())
model_y = make_pipeline(PolynomialFeatures(2), LinearRegression())

# Обучаем
model_x.fit(gaze_samples, points[:, 0])  # gaze → x_px
model_y.fit(gaze_samples, points[:, 1])  # gaze → y_px

# Сохраняем модели
import joblib
joblib.dump(model_x, "calib_x.pkl")
joblib.dump(model_y, "calib_y.pkl")