import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()

        # ===== Свёртки + BatchNorm =====
        self.conv1 = nn.Conv2d(6, 64, kernel_size=5, stride=1, padding=2)
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
        head_pose_tiled = head_pose.view(-1, 3, 1, 1).expand(-1, -1, 60, 36)
        left_input = torch.cat([left_eye, head_pose_tiled], dim=1).float()
        right_input = torch.cat([right_eye, head_pose_tiled], dim=1).float()

        # ===== Левый глаз =====
        x = F.relu(self.bn1(self.conv1(left_input)))
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
        y = F.relu(self.bn1(self.conv1(right_input)))
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