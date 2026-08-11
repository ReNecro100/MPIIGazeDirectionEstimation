from torch.utils.data import Dataset, DataLoader
from PIL import Image
import GetNormalize
import cv2
import scipy.io as sio

import NormalizeFace

class CyrillicLettersDataset(Dataset):
    def __init__(self, img_dir, transform=None, is_train=True):
        #Сделать лист с инфой
        self.learner = []
        six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')
        for participant in range(0,15):
            if participant < 10:
                participant = '0' + str(participant)
            else:
                participant = str(participant)
            path = f"D:\MPIIGaze\MPIIGaze\Annotation Subset\p{participant}.txt"
            camera = sio.loadmat(f'D:\MPIIGaze\MPIIGaze\Data\Original\p{participant}\Calibration\Camera.mat')

            with open(path, "r", encoding="utf-8") as f:
                strinks = [i.split() for i in f.readlines()]

            for i in strinks:
                a = GetNormalize.yes(i, six_point_face, camera)
                b = NormalizeFace.NormalizeFace(six_point_face, 'D:\MPIIGaze\MPIIGaze\Data\Original\p00', a, camera)
                self.learner.append(b)

    def __len__(self):
        return len(self.learner)

    def __getitem__(self, idx):
        left_eye = self.learner[idx]['left_eye']
        right_eye = self.learner[idx]['right_eye']
        euler_angles = self.learner[idx]['euler_angles']
        gaze_vector = self.learner[idx]['gaze_vector']
        # if self.transform:
        #     image = self.transform(image)
        return left_eye, right_eye, euler_angles, gaze_vector