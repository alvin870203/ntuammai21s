"""

"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
# from prettytable import PrettyTable
from torch.utils.data import DataLoader
from test_protocol.utils.model_loader import ModelLoader
from test_protocol.utils.extractor.feature_extractor import CommonExtractor
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory
from data_processor.train_dataset import ImageDataset

import random
import cv2
import numpy as np
from torch.utils.data import Dataset

def accu_key(elem):
    return elem[1]

class myImageDataset(Dataset):
    def __init__(self, data_root, train_file, crop_eye=False):
        self.data_root = data_root
        self.train_list = []
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, image_label = line.split(' ')
            self.train_list.append((image_path, int(image_label)))
            line = train_file_buf.readline().strip()
        self.crop_eye = crop_eye
    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        if self.crop_eye:
            image = image[:60, :]
        #image = cv2.resize(image, (128, 128)) #128 * 128
        # if random.random() > 0.5:
        #     image = cv2.flip(image, 1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
        return image, image_path

# def extract_online(self, model, data_loader):
#     """Extract and return features.
    
#     Args:
#         model(object): initialized model.
#         data_loader(object): load data to be extracted.

#     Returns:
#         image_name2feature(dict): key is the name of image, value is feature of image.
#     """
#     model.eval()
#     image_name2feature = {}
#     with torch.no_grad(): 
#         for batch_idx, (images, filenames) in enumerate(data_loader):
#             images = images.to(self.device)
#             features = model(images)
#             features = F.normalize(features)
#             features = features.cpu().numpy()
#             for filename, feature in zip(filenames, features): 
#                 image_name2feature[filename] = feature
#     return image_name2feature

if __name__ == '__main__':
    # conf = argparse.ArgumentParser(description='apd test protocal.')
    # conf.add_argument("--test_set", type = str, 
    #                   help = "apd, lfw, cplfw, calfw, agedb, rfw_African, \
    #                   rfw_Asian, rfw_Caucasian, rfw_Indian.")
    # conf.add_argument("--data_conf_file", type = str, 
    #                   help = "the path of data_conf.yaml.")
    # conf.add_argument("--backbone_type", type = str, 
    #                   help = "Resnet, Mobilefacenets..")
    # conf.add_argument("--backbone_conf_file", type = str, 
    #                   help = "The path of backbone_conf.yaml.")
    # conf.add_argument('--batch_size', type = int, default = 1024)
    # conf.add_argument('--model_path', type = str, default = 'mv_epoch_8.pt', 
    #                   help = 'The path of model or the directory which some models in.')
    # args = conf.parse_args()
    # parse config.
    # with open(args.data_conf_file) as f:
    #     data_conf = yaml.load(f)[args.test_set]
    #     pairs_file_path = data_conf['pairs_file_path']
    #     cropped_face_folder = data_conf['cropped_face_folder']
    #     image_list_file_path = data_conf['image_list_file_path']
    # # define pairs_parser_factory
    # pairs_parser_factory = PairsParserFactory(pairs_file_path, args.test_set)
    # define dataloader
    # data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False), 
    #                          batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    #model def
    backbone_factory = BackboneFactory('MobileFaceNet', '/home/chihyuan/ntuammai21s/hw1/project/test_protocol/backbone_conf.yaml')
    model_loader = ModelLoader(backbone_factory)
    feature_extractor = CommonExtractor('cuda:0')
    # lfw_evaluator = LFWEvaluator(data_loader, pairs_parser_factory, feature_extractor)
    # if os.path.isdir(args.model_path):
    #     accu_list = []
    #     model_name_list = os.listdir(args.model_path)
    #     for model_name in model_name_list:
    #         if model_name.endswith('.pt'):
    #             model_path = os.path.join(args.model_path, model_name)
    #             model = model_loader.load_model(model_path)
    #             mean, std = lfw_evaluator.test(model)
    #             accu_list.append((os.path.basename(model_path), mean, std))
    #     accu_list.sort(key = accu_key, reverse=True)
    # else:
    model = model_loader.load_model('/home/chihyuan/ntuammai21s/hw1/project/training_mode/conventional_training/out_dir/Epoch_17.pt')
    
    data_loader = DataLoader(myImageDataset('/home/chihyuan/ntuammai21s/hw1/project/data/train/C_prep', '/home/chihyuan/ntuammai21s/hw1/project/data/files/train_list.txt'), 
                             batch_size=128, shuffle=False, num_workers = 4)
    model.eval()
    image_name2feature = feature_extractor.extract_online(model, data_loader)
    print(len(image_name2feature), len(image_name2feature['/home/chihyuan/ntuammai21s/hw1/project/data/train/C_prep/黃義交_5.jpg']))
    print(np.linalg.norm(image_name2feature['/home/chihyuan/ntuammai21s/hw1/project/data/train/C_prep/黃義交_5.jpg']))
    feat1 = image_name2feature['/home/chihyuan/ntuammai21s/hw1/project/data/train/C_prep/黃義交_17.jpg']
    feat2 = image_name2feature['/home/chihyuan/ntuammai21s/hw1/project/data/train/C_prep/黃義交_13.jpg']
    feat3 = image_name2feature['/home/chihyuan/ntuammai21s/hw1/project/data/train/C_prep/青木愛_13.jpg']
    cur_score2 = np.dot(feat1, feat2)
    cur_score3 = np.dot(feat1, feat3)
    print(cur_score2, cur_score3)