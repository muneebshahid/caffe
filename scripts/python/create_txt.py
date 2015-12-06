import numpy as np
import glob 
import os

proj_home = os.getenv('PROJ_HOME')
train_path = proj_home + '/data/train/processed_images/'

train_files = glob.glob(train_path + '*.jpg')
train_txt = proj_home + '/data/train/train.txt'

sorted_files = sorted(train_files)

f = open(train_txt, 'w')
f.writelines(line + ' 1\n' for line in sorted_files)
f.close()

