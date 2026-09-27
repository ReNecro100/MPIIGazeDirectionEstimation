import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()

        # ===== Свёртки + BatchNorm =====
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # ===== Полносвязная часть =====
        # Размер входа: 512 * (60/16) * (36/16) = 512 * 3 * 2 = 3072
        self.fc_eye = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        # Объединение двух глаз + поза
        self.fc1 = nn.Linear(256 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, left_eye, right_eye, head_pose):
        # ===== Поза на входе =====
        #head_pose_tiled = head_pose.view(-1, 3, 1, 1).expand(-1, -1, 60, 36)
        #left_input = torch.cat([left_eye, head_pose_tiled], dim=1).float()
        #right_input = torch.cat([right_eye, head_pose_tiled], dim=1).float()

        # ===== Левый глаз =====
        x = F.relu(self.bn1(self.conv1(left_eye)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc_eye(x)

        # ===== Правый глаз =====
        y = F.relu(self.bn1(self.conv1(right_eye)))
        y = self.pool1(y)
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.pool2(y)
        y = F.relu(self.bn3(self.conv3(y)))
        y = self.pool3(y)
        y = F.relu(self.bn4(self.conv4(y)))
        y = self.pool4(y)
        y = y.view(y.size(0), -1)
        y = self.fc_eye(y)

        # ===== Объединение =====
        combined = torch.cat([x, y], dim=1)
        z = F.relu(self.fc1(combined))
        z = F.relu(self.fc2(z))
        gaze = self.fc3(z)

        # ===== Нормализация выхода =====
        gaze = F.normalize(gaze, p=2, dim=1)
        return gaze


import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # Сверточные слои остаются прежними (они извлекают признаки из каждого глаза)
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Удваиваем входной размер для fc1, так как мы объединяем два вектора по (16 * 5 * 5)
        self.fc1 = nn.Linear(16 * 91 * 2, 120) #tam bylo 5*5 vmesto 91
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, left_eye, right_eye, head_pose):
        '''
        Один проход сети для двух изображений (левый и правый глаз).
        '''
        # 1. Извлекаем признаки из левого глаза
        x1 = F.max_pool2d(F.relu(self.conv1(left_eye.float())), (2, 2))
        x1 = F.max_pool2d(F.relu(self.conv2(x1)), (2, 2))
        x1 = x1.view(x1.size(0), -1)  # Сплющиваем в вектор (размер: 16 * 5 * 5)

        # 2. Извлекаем признаки из правого глаза (используя те же веса)
        x2 = F.max_pool2d(F.relu(self.conv1(right_eye.float())), (2, 2))
        x2 = F.max_pool2d(F.relu(self.conv2(x2)), (2, 2))
        x2 = x2.view(x2.size(0), -1)  # Сплющиваем в вектор (размер: 16 * 5 * 5)

        # 3. Объединяем векторы признаков обоих глаз вместе
        # Склеиваем по размерности 1 (размерность признаков, где dim=0 — это batch_size)
        x = torch.cat((x1, x2), dim=1)

        # 4. Пропускаем через полносвязные слои
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
