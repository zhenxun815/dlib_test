import dlib
import sys
from imageio import imread
import glob
import numpy as np

detector = dlib.get_frontal_face_detector()

predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

image_types = ['*.jpg', '*.jpeg']


def get_feature(path):
    img = imread(path)
    dets = detector(img, 1)

    shape = predictor(img, dets[0])
    face_vector = facerec.compute_face_descriptor(img, shape)
    return face_vector


def distance(a, b):
    a, b = np.array(a), np.array(b)
    sub = np.sum((a - b) ** 2)
    add = (np.sum(a ** 2) + np.sum(b ** 2)) / 2.
    return sub / add


def classifier(a, b, t=0.1):
    diff = distance(a, b)
    print(f'diff is {diff}')
    return diff <= t


def find():
    repo_images=[]
    for type in image_types:
        repo_images.extend(glob.glob(f'./face_repo/{type}'))

    test_face = sys.argv[1]
    print(f'face to test is {test_face}')
    test_feature = get_feature(test_face)

    feature_lists1 = [(get_feature(path), path) for path in repo_images]
    result_face = [path for feature, path in feature_lists1 if classifier(feature, test_feature)]

    for path in result_face:
        print(f'path is {path}')


if __name__ == '__main__':
    find()
