import torch
import torch.nn as nn
import torch.nn.functional as F

class lenet(nn.Module):

    def __init__(self):
        super(lenet, self).__init__()
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