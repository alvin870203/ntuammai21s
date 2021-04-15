"""

"""

import os
import sys
import math
import multiprocessing

import yaml
import cv2
import numpy as np
sys.path.append('./face_sdk')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("face_sdk/config/logging.conf")
logger = logging.getLogger('api')
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

import argparse

with open('face_sdk/config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

def crop_apd(apd_root, target_folder):#, start_idx, end_idx):
    # common setting for all models, need not modify.
    model_path = 'face_sdk/models'

    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Falied to load face detection Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    face_cropper = FaceRecImageCropper()
    file_list = os.listdir(apd_root)
    file_list.sort()
    idx = 0
    for cur_file in file_list:#[start_idx:end_idx]:
        if cur_file.endswith('.jpg'):
            idx += 1
            cur_file_path = os.path.join(apd_root, cur_file)
            cur_image = cv2.imread(cur_file_path)


            dets = faceDetModelHandler.inference_on_image(cur_image)
            face_nums = dets.shape[0]
            if face_nums != 1:
                logger.info('Input image should contain only one face!' + str(face_nums) + ' ' + str(idx))
                # cur_file = str(face_nums) + '-' + cur_file
                # continue
            landmarks = faceAlignModelHandler.inference_on_image(cur_image, dets[0])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cur_cropped_image = face_cropper.crop_image_by_mat(cur_image, landmarks_list)

            target_path = os.path.join(target_folder, cur_file)
            cv2.imwrite(target_path, cur_cropped_image)
        
if __name__ == '__main__':
    # conf = argparse.ArgumentParser(description='lfw test protocal.')
    # conf.add_argument("--batch_idx", type = int, 
    #                   help = "0, 1, ..., 9,")
    # args = conf.parse_args()
    apd_root = '/home/chihyuan/ntuammai21s/hw1/project/data/test/open_set/test_pairs'#'/home/chihyuan/ntuammai21s/hw1/project/data/test/closed_set/test_pairs'
    target_folder = '/home/chihyuan/ntuammai21s/hw1/project/data/test/open_set/test_pairs_crop'#'/home/chihyuan/ntuammai21s/hw1/project/data/test/closed_set/test_pairs_crop'
    # for i in range(9):
    # i = args.batch_idx
    # crop_apd(apd_root, target_folder, 200*i, 200*(i+1))
    crop_apd(apd_root, target_folder)
