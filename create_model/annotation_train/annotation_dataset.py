from torch.utils.data import Dataset
import scipy.io as sio

from create_model.processing.normalize_face import *
from create_model.processing.get_rtvecs import *

class MPII_gaze_dataset(Dataset):
    def __init__(self, is_train=True, write_creation_process=False):
        #Сделать лист с инфой
        self.learner = []
        six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')

        #Для валидации - участник №14, для тренировки - остальные
        if is_train:
            participations_range = range(0, 14)
        else:
            participations_range = [14]

        for participant in participations_range:
            if participant < 10:
                participant = "0"+str(participant)
            else:
                participant = str(participant)
            path = f"D:\MPIIGaze\MPIIGaze\Annotation Subset\p{participant}.txt"
            camera = sio.loadmat(f'D:\MPIIGaze\MPIIGaze\Data\Original\p{participant}\Calibration\Camera.mat')

            with open(path, "r", encoding="utf-8") as f:
                strinks = [i.split() for i in f.readlines()]

            for idx, i in enumerate(strinks):
                if write_creation_process:
                    print(f"p{participant}: {idx}/{len(strinks)}")
                a = get_rtvecs(i, six_point_face, camera)
                b = normalize_face(six_point_face, a, camera, f'D:\MPIIGaze\MPIIGaze\Data\Original\p{participant}')
                self.learner.append(b)

    def __len__(self):
        return len(self.learner)

    def __getitem__(self, idx):
        left_eye = self.learner[idx]['left_eye']
        right_eye = self.learner[idx]['right_eye']
        euler_angles = self.learner[idx]['euler_angles']
        gaze_vector = self.learner[idx]['gaze_vector']  # numpy array
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)  # нормализация через numpy

        return left_eye, right_eye, euler_angles, gaze_vector