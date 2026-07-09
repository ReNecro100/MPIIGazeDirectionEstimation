# #pip install opencv-python mediapipe
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#
# # Загружаем модель
# model_path = r'D:\MPIIGaze\face_landmarker.task'
#
# options = vision.FaceLandmarkerOptions(
#     base_options=python.BaseOptions(model_asset_path=model_path),
#     output_face_blendshapes=True,
#     output_facial_transformation_matrixes=True,
#     num_faces=1,
#     running_mode=vision.RunningMode.IMAGE
# )
#
# image_path = r'D:\MPIIGaze\Alanwilder.png'#r"D:\MPIIGaze\MPIIGaze\Data\Original\p10\day08\0011.jpg"
# frame = cv2.imread(image_path)
#
# if frame is None:
#     print(f"Ошибка: не удалось загрузить изображение '{image_path}'")
#     exit()
#
# # Создаём детектор и обрабатываем изображение
# with vision.FaceLandmarker.create_from_options(options) as landmarker:
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#     detection_result = landmarker.detect(mp_image)
#
# # Рисуем точки
# if detection_result.face_landmarks:
#     img_h, img_w, _ = frame.shape
#     for face_landmarks in detection_result.face_landmarks:
#         for (i, landmark) in enumerate(face_landmarks):
#             print(landmark)
#             if i in [61, 291, 33, 263, 133, 362]:
#                 """Left eye outer corner 33
#                 Right eye outer corner 263
#                 Left eye inner corner 133
#                 Right eye inner corner 362
#                 Mouth left corner 61
#                 Mouth right corner 291"""
#                 x = int(landmark.x * img_w)
#                 y = int(landmark.y * img_h)
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#                 #cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
#
#     cv2.imwrite('output_face_points.jpg', frame)
#     cv2.imshow('Face Points', frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Лицо не обнаружено")

frame = cv2.imread("D:\\MPIIGaze\MPIIGaze\Data\Original\p00\day13\\0203.jpg")

path = "D:\MPIIGaze\MPIIGaze\Annotation Subset\\p00.txt"
with open(path, "r", encoding="utf-8") as f:
    text = f.readline().split()

for i in range(1, len(text)-4, 2):
    print(text[i])
    cv2.circle(frame, (int(text[i]), int(text[i+1])), 2, (0, 255, 0), -1)

cv2.imwrite('output_face_points.jpg', frame)
cv2.imshow('Face Points', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()