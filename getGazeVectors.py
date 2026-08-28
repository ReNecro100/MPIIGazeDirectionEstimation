import NeuroNet
import torch
import dataInference
import scipy.io as sio
from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = NeuroNet.GazeCNN().to(device)
checkpoint = torch.load("D:/MPIIGaze/gaze_vector_finder.pth", weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')

# if is_train:
#     participations_range = range(0, 14)
# else:
#     participations_range = [14]

participations_range = range(0, 14)
counter = 0

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

                counter += 1
                left_eye = torch.tensor(a["left_eye"], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, 60, 36)
                right_eye = torch.tensor(a["right_eye"], dtype=torch.float32).unsqueeze(0).to(device)
                head_pose = torch.tensor(a["euler_angles"], dtype=torch.float32).unsqueeze(0) .to(device) # (1, 3)

                # Подаём в модель
                result = model(left_eye, right_eye, head_pose)

                print(result, counter)