import filesystem_helper as fh
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source_folder_path", help="Path to the source folder (AMOS)")
parser.add_argument("target_folder_path", help="Path to the target folder, where txt file will be created")
args = parser.parse_args()

source_folder_path = args.source_folder_path

valid_camera_nums = np.load('amos/valid_camera_nums.npy')

if not fh.path_exists(source_folder_path):
	print "Invalid source folder path"

folder_contents = fh.get_folder_contents(source_folder_path)
data_set = []
limit_per_folder = 10000
for folder_content in folder_contents:	
	if fh.is_dir(folder_content) and fh.extract_name_from_path(folder_content) in valid_camera_nums:
		camera_folder_path = folder_content + '/'
		camera_num = fh.extract_name_from_path(folder_content)		
		print "Found Amos Camera #: " + camera_num
		camera_folder_contents = fh.get_folder_contents(camera_folder_path)
		months_per_cam = len(camera_folder_contents)
		print "Found Months: " + str(months_per_cam)
		im_added = 0
		if months_per_cam == 1:
			month = camera_folder_contents[0] + '/'
			print "In month: " + month
			images = fh.get_folder_contents(month, '*.jpg')
			total_images = len(images)
			print "Images Found: " + str(total_images)
			while im_added < limit_per_folder:				
				im1, im2 = np.random.random_integers(0, total_images - 1, 2)
				data_set.append([im1, im2, 1])
				im_added = im_added + 1
				if im_added % 500 == 0:
					print 'Added: ' + str(im_added) + ' / ' + str(limit_per_folder)
			break					
