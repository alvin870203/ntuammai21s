"""

"""

import os

def gen_pairs_file(data_root, labels_file, pairs_file):
    """
    Generate the pairs file with same format as CPLFW's pairs file
    """
    pairs_file_buf = open(train_file, 'w')
    image_list = os.listdir(data_root)
    #image_list.sort()
    labels_list = []
    labels_file_buf = open(pairs_file)
    