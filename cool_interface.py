import cv2
import requests
import numpy as np

# Подключение к веб-камере (0 — стандартная камера) и какие-то дефолтные настройки

w, h = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 5)

# Проверка, удалось ли запустить камеру
if not cap.isOpened():
    print("Не удалось открыть камеру")
count = 0
while True:
    # Чтение одного кадра
    ret, frame = cap.read()
    if not ret:
        break
    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    url = " http://127.0.0.1:2137/gaze_vector"

    response = requests.post(url, json = {"frame": frame.tolist()})

    if response.status_code == 200:
        data = response.json()
    else:
        data = {}

    x, y = data["x"], data["y"] #gaze_to_pixel_calibrated(gaze[:2])

    print(f"Точка взгляда: ({x}, {y})")

    cv2.circle(screen, (x, y), 20, (0, 0, 255), -1)  # красный круг
    cv2.putText(screen, f"Gaze: ({x}, {y})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    count += 1
    cv2.imshow("frame", screen)
    cv2.imshow('me', frame)
    if cv2.pollKey() & 0xFF == ord('q'):
        break

    # Показываем

# Обязательно освобождаем ресурс камеры
cap.release()
cv2.destroyAllWindows()