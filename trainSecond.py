import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
import dataset, NeuroNet
from torch.utils.data import DataLoader

val_dst = dataset.MPIIGazesDataset(is_train=False)

val_dataloader = DataLoader(dataset=val_dst, batch_size=32, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_inferenced = NeuroNet.GazeCNN().to(device)
model_inferenced.load_state_dict(torch.load(r"D:/MPIIGaze/gaze_vector_finder_inference.pth"))
criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model_inferenced.parameters(), lr=0.001)

model_inferenced.eval()  # Переключаем в режим оценки (отключаем Dropout)
total_loss = 0
with torch.no_grad():  # Отключаем вычисление градиентов (экономит память)
    for left, right, pose, target in val_dataloader:
        left, right, pose, target = left.to(device), right.to(device), pose.to(device), target.to(device)

        output = model_inferenced(left, right, pose)
        val_loss = criterion(output, target, torch.ones(output.size(0)).to(device))
        total_loss += val_loss.item()  # Суммируем потери

avg_loss = total_loss / len(val_dataloader)  # Средняя ошибка на валидации
print(avg_loss)

model_before = NeuroNet.GazeCNN().to(device)
model_before.load_state_dict(torch.load(r"D:/MPIIGaze/gaze_vector_finder.pth"))
criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model_before.parameters(), lr=0.001)

model_before.eval()  # Переключаем в режим оценки (отключаем Dropout)
total_loss = 0
with torch.no_grad():  # Отключаем вычисление градиентов (экономит память)
    for left, right, pose, target in val_dataloader:
        left, right, pose, target = left.to(device), right.to(device), pose.to(device), target.to(device)

        output = model_before(left, right, pose)
        val_loss = criterion(output, target, torch.ones(output.size(0)).to(device))
        total_loss += val_loss.item()  # Суммируем потери

avg_loss = total_loss / len(val_dataloader)  # Средняя ошибка на валидации
print(avg_loss)