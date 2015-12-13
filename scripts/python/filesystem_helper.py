import shutil
import os
import glob
import ntpath

def path_exists(path):
	return os.path.exists(path)

def copy(source_path, target_path):
	shutil.copy(source_path, target_path)

def rm_dir(path):
	shutil.rmtree(path)

def make_dir(path):
	os.makedirs(path)

def get_folder_contents(path, wild_card=''):
	return glob.glob(path + wild_card)

def get_file_name(path):
	return ntpath.basename(path)	