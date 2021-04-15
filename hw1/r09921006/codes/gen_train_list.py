"""
@author: Chih-Yuan Chuang
@date: 20210330
@contact: r09921006@g.ntu.edu.tw
"""

import os

def gen_train_list(data_root, train_file, data_root_unlabeled):
    """
    Generate the train list file, which has the following format.

    relative_path0 label0
    relative_path1 label1
    relative_path2 label2
    """
    train_file_buf = open(train_file, 'w')
    image_list = os.listdir(data_root)
    image_list.sort()
    cur_label_name = '' # current <Name of person>
    cur_label = -1 # current <Index number of person>
    for index, image_path in enumerate(image_list):
        label_name = image_path.split('_')[0] # ['<Name of person>', '<number>.jpg']
        if label_name != cur_label_name:
            cur_label_name = label_name
            cur_label += 1
        #print(image_path, cur_label)
        line = image_path + ' ' + str(cur_label)
        train_file_buf.write(line + '\n')
    
    image_list_unlabeled = os.listdir(data_root_unlabeled)
    for index, image_path in enumerate(image_list_unlabeled):
        cur_label += 1
        #print(image_path, cur_label)
        line = '../../data_test/open_set/unlabeled_data_prep/' + image_path + ' ' + str(cur_label)
        train_file_buf.write(line + '\n')


if __name__ == '__main__':
    data_root = './data/train/C_prep'
    data_root_unlabeled = './data/test/open_set/unlabeled_data_prep'
    # file to be generate.
    train_file = './data/files/train/train_list_unsupervised.txt'
    gen_train_list(data_root, train_file, data_root_unlabeled)