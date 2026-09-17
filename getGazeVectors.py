import NeuroNet
import torch
from torch import nn
import dataInference
import scipy.io as sio
from pathlib import Path

import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = NeuroNet.GazeCNN().to(device)
checkpoint = torch.load("D:/MPIIGaze/gaze_vector_finder.pth", weights_only=True)
model.load_state_dict(checkpoint)
model.train()

criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')
counter = 0
batch_left, batch_right, batch_pose, batch_gaze = [], [], [], []
BATCH_SIZE = 32
min_loss = 0.05

# if is_train:
#     participations_range = range(0, 14)
# else:
#     participations_range = [14]

participations_range = range(0, 15)

for participant in participations_range:
    if participant < 10:
        participant = "0" + str(participant)
    else:
        participant = str(participant)
    camera = sio.loadmat(rf'D:\MPIIGaze\MPIIGaze\Data\Original\p{participant}\Calibration\Camera.mat')
    target_dir = Path(rf"D:\MPIIGaze\MPIIGaze\Data\Original\p{participant}")

    for dr in [x for x in target_dir.iterdir() if x.is_dir() and x.name[-11:] != "Calibration"]:
        with open(str(dr) + r"\annotation.txt", "r", encoding="utf-8") as f:
            annotations = f.readlines()
        for path in dr.iterdir():
            if path.is_file() and path.name.endswith(".jpg"):
                annotation_num = int(path.name[-8:][:4]) - 1
                a = dataInference.NormalizeFaceInference(six_point_face, str(dr)+'\\' +path.name, camera, annotations[annotation_num])

                left_eye = a["left_eye"]
                right_eye = a["right_eye"]
                head_pose = a["euler_angles"]

                # cv2.namedWindow("left_eye", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("left_eye", 60, 36)  # ← точный размер
                # cv2.imshow('left_eye', left_eye.transpose(2, 1, 0))
                #
                # cv2.namedWindow('right_eye', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('right_eye', 60, 36)  # ← точный размер
                # cv2.imshow('right_eye', right_eye.transpose(2, 1, 0))
                # print(cv2.getWindowImageRect("left_eye"), left_eye.transpose(2, 1, 0).shape)
                # cv2.waitKey(0)

                # Истинный gaze_vector (из аннотации)
                gaze_vector = torch.tensor([
                    float(annotations[annotation_num].split()[26]),
                    float(annotations[annotation_num].split()[27]),
                    float(annotations[annotation_num].split()[28])
                ], dtype=torch.float32)
                gaze_vector = gaze_vector / torch.norm(gaze_vector)  # нормализация

                # Накапливаем в батч
                batch_left.append(left_eye)
                batch_right.append(right_eye)
                batch_pose.append(head_pose)
                batch_gaze.append(gaze_vector)

                counter += 1

                # Когда набралось 32 — обучаем
                if len(batch_left) == BATCH_SIZE:
                    left_batch = torch.tensor(np.array(batch_left), dtype=torch.float32).to(device)
                    right_batch = torch.tensor(np.array(batch_right), dtype=torch.float32).to(device)
                    pose_batch = torch.tensor(np.array(batch_pose), dtype=torch.float32).to(device)
                    gaze_batch = torch.tensor(np.array(batch_gaze), dtype=torch.float32).to(device)

                    result = model(left_batch, right_batch, pose_batch)
                    loss = criterion(result, gaze_batch, torch.ones(result.size(0)).to(device))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_left, batch_right, batch_pose, batch_gaze = [], [], [], []
                    del left_batch, right_batch, pose_batch, gaze_batch, result
                    print(f"{counter}/213659, Потеря: {round(loss.item(), 5)}, Минимум: {min_loss}")
                    if counter > 150_000 and loss.item() < min_loss:
                        min_loss = round(loss.item(), 5)
                        torch.save(model.state_dict(), "D:/MPIIGaze/gaze_vector_finder_inference.pth")
                        print("\n!!!Сохранил!!!\n")

