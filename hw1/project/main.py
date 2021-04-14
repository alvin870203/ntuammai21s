"""

"""

import os
import argparse
from training_mode.conventional_training.train import train_main
from test_protocol.test_lfw import test_main
from prep_face import prep_train_data
from gen_train_list import gen_train_list
from gen_image_list import gen_image_list
from gen_pairs_file import gen_pairs_file

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='main code for face recognition project.')
    conf.add_argument("--mode", type=str,
                      help = "train/closed/open")
    conf.add_argument("--head_type", type = str, default='AM-Softmax',
                      help = "mv-softmax, arcface, npc-face.")
    # conf.add_argument("--test_set", type = str, 
    #                   help = "lfw, cplfw, calfw, agedb, rfw_African, \
    #                   rfw_Asian, rfw_Caucasian, rfw_Indian.")
    
    ### use default value ###
    conf.add_argument("--data_conf_file", type = str, default='./test_protocol/data_conf.yaml',
                      help = "the path of data_conf.yaml.")
    conf.add_argument('--model_path', type = str, default = './training_mode/conventional_training/out_dir/Epoch_184.pt', 
                      help = 'The path of model or the directory which some models in.')

    ### use default value ###
    conf.add_argument("--data_root", type = str, default='./data/train/C_prep', 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str, default='./data/files/train/train_list_unsupervised.txt', 
                      help = "The training file path.")
    conf.add_argument("--backbone_type", type = str, default='MobileFaceNet', 
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str, default='./training_mode/backbone_conf.yaml', 
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_conf_file", type = str, default='./training_mode/head_conf.yaml', 
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, default='./training_mode/conventional_training/out_dir', 
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 200, 
                      help = 'The training epoches.')
    conf.add_argument('--step', type = str, default = '80, 140, 180', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 200, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 3000, 
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128, 
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type = float, default = 0.9, 
                      help = 'The momentum for sgd.')
    conf.add_argument('--log_dir', type = str, default = 'training_mode/conventional_training/log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, default='mv-hrnet', 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Whether to resume from a checkpoint.')
    ######
    args = conf.parse_args()
    if args.mode == 'train':
        prep_train_data('./data/train/C', './data/train/C_prep')
        gen_train_list('./data/train/C_prep', './data/files/train/train_list_unsupervised.txt', './data/test/open_set/unlabeled_data_prep')
        train_main(args)
    if args.mode == 'closed':
        gen_image_list('./data/test/closed_set/test_pairs_crop', './data/files/test/closed_set/img_list_closed.txt')
        gen_pairs_file('./data/test/closed_set/test_pairs_crop', './data/test/closed_set/labels.txt', './data/files/test/closed_set/pairs_closed.txt')
        args.test_set = 'APD'
        test_main(args)
    if args.mode == 'open':
        gen_image_list('./data/test/open_set/test_pairs_crop', './data/files/test/open_set/img_list_open.txt')
        gen_pairs_file('./data/test/open_set/test_pairs_crop', './data/test/open_set/labels.txt', './data/files/test/open_set/pairs_open.txt')
        args.test_set = 'OPEN'
        test_main(args)
