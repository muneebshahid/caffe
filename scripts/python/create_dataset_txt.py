import os_helper as osh
import argparse
import sys
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("source_folder_path", help="Path to the training source folder")
parser.add_argument("target_folder_path", help="Path to the target folder, where txt file will be created")
args = parser.parse_args()

folder_path_mich = args.source_folder_path + 'michigan/uncompressed/'
folder_path_amos = args.source_folder_path + 'amos/uncompressed/'

#Process Michigan data
months_mich = ['aug', 'jan']
data_set_michigun = []
files_michigan_pos = sorted(osh.get_folder_contents(folder_path_mich +'jan/', '*.tiff'))
files_michigan_pos_len = len(files_michigan_pos)
for jan_file in files_michigan_pos:
	jan_file = jan_file.replace(args.source_folder_path, '')

	month_mich = np.random.random_integers(0, 1)
	path_1 = jan_file.replace('jan', months_mich[month_mich])
	path_2 = jan_file.replace('jan', months_mich[abs(month_mich - 1)])
	#label_1 is simillarity label, while label_2 is the domain_label
	label_1, label_2 = 1, 1	
	data_set_michigun.append([path_1, path_2, label_1, label_2])

images_gap = 600
michigan_neg_instances = 0
while michigan_neg_instances < files_michigan_pos_len:
	
	im1 = np.random.random_integers(0, files_michigan_pos_len - 1)
	month_mich = np.random.random_integers(0, 1)
	file_1 = folder_path_mich + months_mich[month_mich] + '/00000' + str(im1) + '.tiff'

	#michigan dataset has weird naming convetion
	if not osh.is_file(file_1):
		continue

	#ensure imagegap between negative examples
	im_diff = im1 - images_gap

	if im_diff > 0:
		im2 = np.random.random_integers(0, im_diff)
	else:
		im2 = np.random.random_integers(im1, im1 + images_gap + 1)
	
	file_2 = folder_path_mich + months_mich[abs(month_mich - 1)] + '/00000' + str(im2) + '.tiff'
	if not osh.is_file(file_2):
		continue
	file_1 = file_1.replace(folder_path_mich, 'michigan/uncompressed/')
	file_2 = file_2.replace(folder_path_mich, 'michigan/uncompressed/')
	label_1, label_2 = 0, 1
	data_set_michigun.append([file_1, file_2, label_1, label_2])
	michigan_neg_instances = michigan_neg_instances + 1

del files_michigan_pos
random.shuffle(data_set_michigun)

with open(args.target_folder_path + '/train1.txt', 'w') as t1:
	t1.writelines([str(instance[0]) + ' ' + str(instance[2]) + '\n' for instance in data_set_michigun])

with open(args.target_folder_path + '/train2.txt', 'w') as t2:
	t2.writelines([str(instance[1]) + ' ' + str(instance[3]) + '\n' for instance in data_set_michigun])
'''


#Process Amos data
valid_camera_nums = np.load('amos/valid_camera_nums.npy')
if not osh.path_exists(folder_path_amos):
	print "Invalid source folder path"

folder_contents = osh.get_folder_contents(folder_path_amos)
data_set = []
limit_per_folder = 10000
for folder_content in folder_contents:	
	if osh.is_dir(folder_content) and osh.extract_name_from_path(folder_content) in valid_camera_nums:
		camera_folder_path = folder_content + '/'
		camera_num = osh.extract_name_from_path(folder_content)		
		print "Found Amos Camera #: " + camera_num
		camera_folder_contents = osh.get_folder_contents(camera_folder_path)
		months_per_cam = len(camera_folder_contents)
		print "Found Months: " + str(months_per_cam)
		im_added = 0
		if months_per_cam == 1:
			month = camera_folder_contents[0] + '/'
			print "In month: " + month
			images = osh.get_folder_contents(month, '*.jpg')
			total_images = len(images)
			print "Images Found: " + str(total_images)
			while im_added < limit_per_folder:				
				im1, im2 = np.random.random_integers(0, total_images - 1, 2)
				data_set.append([im1, im2, 1])
				im_added = im_added + 1
				if im_added % 500 == 0:
					print 'Added: ' + str(im_added) + ' / ' + str(limit_per_folder)
			break					
'''