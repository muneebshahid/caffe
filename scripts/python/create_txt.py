import numpy as np
import os_helper as osh

proj_home = osh.get_env_var('PROJ_HOME')
path_michigan = proj_home + '/data/train/michigan/uncompressed/'
path_amos = proj_home + '/data/train/amos/'


#Process michigun
files_jan_michigan = osh.list_dir(path_michigan +'/jan/')
files_aug_michigan = osh.list_dir(path_michigan + '/aug/')

assert files_aug_michigan == files_jan_michigan
files_michigan = np.hstack((np.array(files_aug_michigan)[:, np.newaxis], np.array(files_jan_michigan)[:, np.newaxis]))
positive_examples = files_michigan
positive_examples_len = len(positive_examples)

neg_examples = 0
while neg_examples < positive_examples_len:
	



train_txt = proj_home + '/data/train/train.txt'
			
#f = open(train_txt, 'w')
#f.writelines(line + ' 1\n' for line in sorted_files)
#f.close()

