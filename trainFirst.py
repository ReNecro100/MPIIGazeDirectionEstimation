import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
import dataset, NeuroNet
from torch.utils.data import DataLoader

train_dst = dataset.MPIIGazesDataset(write_creation_process=True)
print(len(train_dst))
val_dst = dataset.MPIIGazesDataset(is_train=False)

train_dataloader = DataLoader(dataset=train_dst, batch_size=32, shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_dst, batch_size=32, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = NeuroNet.GazeCNN().to(device)
print(sum([p.numel() for p in model.parameters() if p.requires_grad]))

criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

epochs = 10
val_acc_history = []
train_acc_history = []

loss = 0
for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    print(f"\n№{epoch + 1}/{epochs}: GO")
    for i, (left_eye, right_eye, euler_angles, gaze_vector) in tqdm(enumerate(train_dataloader)):
        #print(left_eye, right_eye, euler_angles, gaze_vector)
        left_eye, right_eye = left_eye.to(device), right_eye.to(device)
        euler_angles, gaze_vector = euler_angles.to(device), gaze_vector.to(device)
        #For linear models
        # images = torch.mean(images, dim=1, keepdim=False)
        # images = torch.flatten(images, start_dim=1)
        # For linear models
        output = model(left_eye, right_eye, euler_angles)
        loss = criterion(output, gaze_vector, torch.ones(output.size(0)).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()  # Переключаем в режим оценки (отключаем Dropout)
    total_loss = 0
    with torch.no_grad():  # Отключаем вычисление градиентов (экономит память)
        for left, right, pose, target in val_dataloader:
            left, right, pose, target = left.to(device), right.to(device), pose.to(device), target.to(device)

            output = model(left, right, pose)
            val_loss = criterion(output, target, torch.ones(output.size(0)).to(device))
            total_loss += val_loss.item()  # Суммируем потери
            if epoch == 9:
                print(output[2], target[2])

    avg_loss = total_loss / len(val_dataloader)  # Средняя ошибка на валидации
    scheduler.step(avg_loss)

    val_acc_history.append(avg_loss)
    train_acc_history.append(loss.item())
    print(f'Validation: {avg_loss}\nTrain: {loss.item()}')

torch.save(model.state_dict(), "D:/MPIIGaze/gaze_vector_finder.pth")

plt.plot(val_acc_history, color='blue', marker='o', markersize=7, label='Validation')
plt.plot(train_acc_history, color='green', marker='o', markersize=7, label='Training')
plt.legend()
plt.show()