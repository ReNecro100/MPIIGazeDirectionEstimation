import GetNormalize
import cv2
import scipy.io as sio

import NormalizeFace

six_point_face = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\6 points-based face model.mat')
path = r"D:\MPIIGaze\MPIIGaze\Annotation Subset\p00.txt"
camera = sio.loadmat(r'D:\MPIIGaze\MPIIGaze\Data\Original\p00\Calibration\Camera.mat')

a = GetNormalize.yes(six_point_face, path, camera)

print(a[0])

NormalizeFace.NormalizeFace(six_point_face, 'D:\MPIIGaze\MPIIGaze\Data\Original\p00', a, camera)