"""

"""

import os

def gen_pairs_file(data_root, labels_file, pairs_file):
    """
    Generate the pairs file with same format as CPLFW's pairs file
    """
    pairs_file_buf = open(pairs_file, 'w')
    image_list = os.listdir(data_root)
    #image_list.sort()
    #labels_list = []
    with open(pairs_file) as f:
        labels_list = [line.rstrip('\n') for line in f]
    for index, image_path in enumerate(image_list):
        labels_idx = image_path.split('_')[2]
        line = image_path + ' ' + labels_list[labels_idx]
        pairs_file_buf.write(line + '\n')


if __name__ == '__main__':
    data_root = './data/test/closed_set/test_pairs'
    labels_file = './data/test/closed_set/labels.txt'
    # file to be generate.
    pairs_file = './data/files/test/closed_set/pairs_closed.txt'
    gen_pairs_file(data_root, labels_file, pairs_file)
    