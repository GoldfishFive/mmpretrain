import glob
import os
from shutil import move
from os import rmdir

def convert_val():

    target_folder = '/home/user01/datasets/tiny-imagenet-200/val/'

    val_dict = {}
    with open('/home/user01/datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob('/home/user01/datasets/tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = target_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    rmdir('/home/user01/datasets/tiny-imagenet-200/val/images')

def get_label():
    root_dir =  '/home/user01/datasets/tiny-imagenet-200/'
    words_file = os.path.join(root_dir, "words.txt")
    wnids_file = os.path.join(root_dir, "wnids.txt")

    set_nids = set()
    with open(wnids_file, 'r') as fo:
        data = fo.readlines()
        for entry in data:
            set_nids.add(entry.strip("\n"))

    class_to_label = {}
    with open(words_file, 'r') as fo:
        data = fo.readlines()
        for entry in data:
            words = entry.split("\t")
            if words[0] in set_nids:
                class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
                print('"'+class_to_label[words[0]] +'",')
    print(len(class_to_label))
if __name__ == "__main__":
    get_label()