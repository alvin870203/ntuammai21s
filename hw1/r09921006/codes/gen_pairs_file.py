"""

"""

import os

def gen_pairs_file(data_root, labels_file, pairs_file):
    """
    Generate the pairs file with same format as CPLFW's pairs file
    """
    pairs_file_buf = open(pairs_file, 'w')
    image_list = os.listdir(data_root)
    image_list.sort(key= lambda x: int(x.split('_')[3].strip('.jpg')))
    image_list.sort(key= lambda x: int(x.split('_')[2]))
    with open(labels_file) as f:
        labels_list = [line.rstrip('\n') for line in f]
    #print(labels_list)
    for index, image_path in enumerate(image_list):
        labels_idx = int(image_path.split('_')[2])
        #print(image_path, labels_idx)
        line = image_path + ' ' + labels_list[labels_idx]
        pairs_file_buf.write(line + '\n')
    print("Number of lines: {}".format(len(image_list)))


if __name__ == '__main__':
    data_root = './data/test/open_set/test_pairs_crop'#'./data/test/closed_set/test_pairs_crop'
    labels_file = './data/test/open_set/labels.txt'#'./data/test/closed_set/labels.txt'
    # file to be generate.
    pairs_file = './data/files/test/open_set/pairs_open.txt'#'./data/files/test/closed_set/pairs_closed.txt'
    gen_pairs_file(data_root, labels_file, pairs_file)
    