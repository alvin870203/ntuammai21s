"""

"""

import os

def gen_image_list(data_root, image_list_file):
    """

    """
    image_list_file_buf = open(image_list_file, 'w')
    image_list = os.listdir(data_root)
    for index, image_path in enumerate(image_list):
        image_list_file_buf.write(image_path + '\n')
    print("Numbers of images:{}".format(len(image_list)))


if __name__ == '__main__':
    data_root = './data/test/open_set/test_pairs_crop'#'./data/test/closed_set/test_pairs_crop'
    # file to be generate.
    image_list_file = './data/files/test/open_set/img_list_open.txt'#'./data/files/test/closed_set/img_list_closed.txt'
    gen_image_list(data_root, image_list_file)