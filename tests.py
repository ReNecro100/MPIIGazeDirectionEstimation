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

a = 0
for i in [-0.7224753,   0.27920786,  0.63251275]:
    a += i*i
    print(i*i)
print(a)
