import NeuroNet
import torch
from torch import nn
import dataInference
import scipy.io as sio
from pathlib import Path

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

                # Истинный gaze_vector (из аннотации)
                gaze_vector = torch.tensor([
                    float(annotations[annotation_num].split()[26]),
                    float(annotations[annotation_num].split()[27]),
                    float(annotations[annotation_num].split()[28])
                ], dtype=torch.float32).unsqueeze(0).to(device)
                gaze_vector = gaze_vector / torch.norm(gaze_vector)  # нормализация

                # Подаём в модель
                result = model(
                    torch.tensor(left_eye, dtype=torch.float32).unsqueeze(0).to(device),
                    torch.tensor(right_eye, dtype=torch.float32).unsqueeze(0).to(device),
                    torch.tensor(head_pose, dtype=torch.float32).unsqueeze(0) .to(device)
                )
                loss = criterion(result, gaze_vector, torch.ones(result.size(0)).to(device))

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                print(f"{counter}/213659, Потеря: {round(loss.item(), 3)}")
                del left_eye, right_eye, head_pose, result
torch.save(model.state_dict(), "D:/MPIIGaze/gaze_vector_finder_inference.pth")