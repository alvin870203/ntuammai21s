"""
insightface/deploy/face_model.py
"""

import os
import os.path as osp
import shutil
import cv2
from utils import face_align


def get_input(face_img):
    # bbox, pts5 = detector.detect(face_img, threshold=0.8)
    # if bbox.shape[0]==0:
    #     print('No bbox detected')
    #     return None
    # bbox = bbox[0, 0:4]
    # pts5 = pts5[0, :]
    # nimg = face_align.norm_crop(face_img, pts5)
    nimg = face_align.norm_crop(face_img)
    return nimg

def prep_train_data(org_path, tgt_path):
    """
    Step1: Prepare the training data
    Align the face images to 112*112 according to face_align.py.
    """
    if os.path.isdir(tgt_path):
        shutil.rmtree(tgt_path)
    os.mkdir(tgt_path)

    for image_file in os.listdir(org_path):
        img = cv2.imread(osp.join(org_path, image_file))
        img = get_input(img)
        cv2.imwrite(osp.join(tgt_path, image_file), img)

if __name__ == '__main__':
    org_path = './data/train/C'
    tgt_path = './data/train/C_prep'
    prep_train_data(org_path, tgt_path)