# import torch
#
# # Проверка доступности GPU
# print("CUDA доступна:", torch.cuda.is_available())
#
# # Вывод названия вашей видеокарты
# if torch.cuda.is_available():
#     print("Ваша видеокарта:", torch.cuda.get_device_name(0))
#
# import matplotlib.pyplot as plt
# import math
# import scipy.io as sio
# import numpy as np
#
# six_point_face = sio.loadmat('D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\\Camera.mat')
# print(six_point_face)
#
# path = "D:\\MPIIGaze\MPIIGaze\Data\Original\p00\day01\\annotation.txt"
# with open(path, "r", encoding="utf-8") as f:
#     text = f.readline().split()
#
# xs = []
# ys = []
#
# for i in range(0,24,2):
#     xs.append(float(text[i]))
#     ys.append(float(text[i+1]))
#
# xs.append(float(text[24]))
# ys.append(float(text[25]))
#
# camera_data = sio.loadmat('D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\\Camera.mat')
# right_eye = camera_data['cameraMatrix'] @ np.array([float(text[35])/float(text[37]),float(text[36])/float(text[37]),1])
# xs.append(right_eye[0])
# ys.append(right_eye[1])
#
# camera_data = sio.loadmat('D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\\Camera.mat')
# left_eye = camera_data['cameraMatrix'] @ np.array([float(text[38])/float(text[40]),float(text[39])/float(text[40]),1])
# xs.append(left_eye[0])
# ys.append(left_eye[1])
#
# #plt.axis([0,1280,0,720])
# plt.title('My first plot')
# plt.plot(xs,ys,'ro')
# plt.show()
import torch

print(r"D:\MPIIGaze\MPIIGaze\Data\Original\p02\day02\0007.jpg"[-8:][:4]) #0007.jpg
print(r"D:\MPIIGaze\MPIIGaze\Data\Original\p02\day02\0007.jpg"[:-8]) #ОСТАЛЬНОЕ

# mat_data = sio.loadmat('D:\MPIIGaze\MPIIGaze\\6 points-based face model.mat')
# print(mat_data.get('model'))
# greek_list_x, greek_list_y = [], []
# for i in mat_data['model']:
#     length = math.sqrt(float(i[0])**2 + float(i[1])**2 + float(i[2])**2)
#     x = float(i[0])/length
#     y = float(i[1])/length
#     z = float(i[2])/length
#
#     theta = math.asin(-y) #math.asinh(-float(i[1]))
#     phi = math.atan2(-x, -z)
#
#     greek_list_x.append(100+theta*math.cos(phi))
#     greek_list_y.append(100+theta*math.sin(phi))
#
#     length = math.sqrt(float(i[3]) ** 2 + float(i[4]) ** 2 + float(i[5]) ** 2)
#     x = float(i[3]) / length
#     y = float(i[4]) / length
#     z = float(i[5]) / length
#
#     theta = math.asin(-y)  # math.asinh(-float(i[1]))
#     phi = math.atan2(-x, -z)
#
#     greek_list_x.append(100 + theta * math.cos(phi))
#     greek_list_y.append(100 + theta * math.sin(phi))
#
# print(greek_list_x, greek_list_y)
#
# plt.axis([0,1280,0,720])
# plt.title('6p')
# plt.plot(greek_list_x, greek_list_y,'ro')
# plt.show()

# a = 0
# for i in [-0.7224753,   0.27920786,  0.63251275]:
#     a += i*i
#     print(i*i)
# print(a)
#
# import cv2
#
# # 1. Initialize the camera stream or video file
# cap = cv2.VideoCapture(0)  # Use 'video.mp4' for a file
#
# if not cap.isOpened():
#     print("Error: Could not open video source.")
#     exit()
#
# while True:
#     # 2. Read the next frame sequentially
#     ret, frame = cap.read()
#
#     # Break the loop if the video ends or the camera fails
#     if not ret:
#         print("Failed to grab frame or reached end of video.")
#         break
#
#     # --- Your frame processing operations go here ---
#     # Example: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 3. Display the updated frame in a window
#     cv2.imshow('Live Stream', frame)
#
#     # 4. CRITICAL: Wait at least 1ms to process GUI window events
#     # This also allows you to exit the loop by pressing the 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 5. Clean up and release system resources
# cap.release()
# cv2.destroyAllWindows()

# import scipy.io as sio
# import NormalizeFace
# import dataInference
# import GetNormalize
# import cv2
#
# six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')
# with open("D:\MPIIGaze\MPIIGaze\Annotation Subset\p05.txt", "r") as f:
#     annotations = f.readlines()
#
# for annotation in annotations:
#     image_path="D:\MPIIGaze\MPIIGaze\Data\Original\p05\\" + annotation.split(" ")[0]
#     with open(f"D:\MPIIGaze\MPIIGaze\Data\Original\p05\\{annotation.split(" ")[0][:5]}\\annotation.txt", "r") as f:
#         for i in range(int(annotation.split(" ")[0][-8:][:4])):
#             annotation_line = f.readline()
#     camera = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\Data\Original\p05\Calibration\Camera.mat')
#
#     b1 = dataInference.NormalizeFaceInference(six_point_face, image_path, camera, annotation_line)
#
#     # print("=== NormalizeFaceInference ===")
#     print(f"left_eye: {b1['left_eye'].shape}")
#     print(f"right_eye: {b1['right_eye'].shape}")
#     print(f"euler_angles: {b1['euler_angles'].flatten()}")
#
#     # cv2.imshow("Inference: Left Eye", b1["left_eye"].transpose(1, 2, 0))
#     # cv2.imshow("Inference: Right Eye", b1["right_eye"].transpose(1, 2, 0))
#
#     image_path_not_full="D:\MPIIGaze\MPIIGaze\Data\Original\p05"
#     anl = annotation
#
#     # Сначала подготовь rtvecs через GetNormalize.yes
#
#     cv2.imshow("Real: Left Eye", cv2.imread(image_path))
#     cv2.waitKey(0)
#
#     #rtvecs = GetNormalize.give_this_function_a_proper_name_someday(image_path, six_point_face, camera)
#     print("=== MediaPipe ===")
#     #print(rtvecs["rotation_vector"], rtvecs["translation_vector"])
#
#     print("=== Annotation ===")
#     rtvecs = GetNormalize.yes(anl.split(" "), six_point_face, camera)
#     print(rtvecs["rotation_vector"], rtvecs["translation_vector"])
#
#     # print("=== yes ===")
#     # print(f"rotation_vector: {rtvecs['rotation_vector'].flatten()}")
#     # print(f"translation_vector: {rtvecs['translation_vector'].flatten()}")
#
#     b2 = NormalizeFace.NormalizeFace(six_point_face, rtvecs, camera, path=image_path_not_full)
#
#     print("=== NormalizeFace ===")
#     print(f"left_eye: {b2['left_eye'].shape}")
#     print(f"right_eye: {b2['right_eye'].shape}")
#     print(f"euler_angles: {b2['euler_angles'].flatten()}")
#
#     cv2.imshow("Real: Left Eye", b2['left_eye'].transpose(2, 1, 0))
#     cv2.imshow("ffff: Left Eye", b1['left_eye'].transpose(2, 1, 0))
#     cv2.waitKey(0)
#
#     print('\n\n\n')

# Загрузи СТАРУЮ модель
