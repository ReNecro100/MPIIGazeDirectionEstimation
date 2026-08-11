import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeCNN(nn.Module):
    """
    Multimodal CNN for gaze estimation.
    Input: eye images + head pose
    Output: gaze direction
    """

    def __init__(self):
        super(GazeCNN, self).__init__()

        # ===== Свёрточная часть (обрабатывает каждый глаз) =====
        # По инструкции: 2 свёрточных слоя
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)  # уменьшаем размер в 2 раза

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # уменьшаем размер в 2 раза

        # ===== Полносвязная часть для глаз =====
        # Размер после свёрток для входа 60×36:
        # после pool1: 30×18, после pool2: 15×9
        self.fc_eye = nn.Linear(64 * 15 * 9, 128)
        self.dropout = nn.Dropout(0.5)

        # ===== Объединение признаков глаз + поза головы =====
        # Поза головы — 3 числа (pitch, yaw, roll)
        self.fc1 = nn.Linear(128 * 2 + 3, 64)  # *2 — два глаза, +3 — поза
        self.fc2 = nn.Linear(64, 2)  # на выходе 2 числа: gaze_pitch, gaze_yaw

    def forward(self, left_eye, right_eye, head_pose):
        """
        Args:
            left_eye:  (batch, 3, 60, 36)
            right_eye: (batch, 3, 60, 36)
            head_pose: (batch, 3)  — [pitch, yaw, roll]
        """
        # ===== Обработка левого глаза =====
        x = F.relu(self.conv1(left_eye))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc_eye(x))

        # ===== Обработка правого глаза (те же веса) =====
        y = F.relu(self.conv1(right_eye))
        y = self.pool1(y)
        y = F.relu(self.conv2(y))
        y = self.pool2(y)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc_eye(y))

        # ===== Объединение =====
        combined = torch.cat([x, y, head_pose], dim=1)  # (128+128+3) = 259
        z = F.relu(self.fc1(combined))
        z = self.dropout(z)
        gaze = self.fc2(z)

        return gaze