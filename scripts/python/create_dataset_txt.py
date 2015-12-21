import os_helper as osh
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("source_folder_path", help="Path to the source folder")
parser.add_argument("target_folder_path",
                    help="Path to the target folder, where txt file will be created")
parser.add_argument("process_mich", help="Process michigan dataset", type=int)
parser.add_argument("process_freiburg", help="Process freiburg dataset", type=int)
args = parser.parse_args()

folder_path_mich = args.source_folder_path + 'michigan/uncompressed/'
folder_path_amos = args.source_folder_path + 'amos/uncompressed/'
folder_path_frei = args.source_folder_path + 'freiburg/'

if args.process_mich:
    # Process Michigan data
    months_mich = ['aug', 'jan']
    data_set_michigan = []
    files_michigan_pos = sorted(osh.get_folder_contents(folder_path_mich + 'jan/', '*.tiff'))
    files_michigan_pos_len = len(files_michigan_pos)
    for jan_file in files_michigan_pos:
        jan_file = jan_file.replace(args.source_folder_path, '')

        month_mich = np.random.random_integers(0, 1)
        path_1 = jan_file.replace('jan', months_mich[month_mich])
        path_2 = jan_file.replace('jan', months_mich[abs(month_mich - 1)])
        # label_1 is simillarity label, while label_2 is the domain_label
        label_1, label_2 = 1, 1
        data_set_michigan.append([path_1, path_2, label_1, label_2])

    images_gap = 600
    michigan_neg_instances = 0
    while michigan_neg_instances < files_michigan_pos_len:
        im1 = np.random.random_integers(0, files_michigan_pos_len - 1)
        month_mich = np.random.random_integers(0, 1)
        file_1 = folder_path_mich + months_mich[month_mich] + '/00000' + str(im1) + '.tiff'
        # michigan data set has weird naming convetion
        if not osh.is_file(file_1):
            continue
        # ensure image gap between negative examples
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
        data_set_michigan.append([file_1, file_2, label_1, label_2])
        michigan_neg_instances += 1

    del files_michigan_pos
    random.shuffle(data_set_michigan)

    with open(args.target_folder_path + '/train1.txt', 'w') as t1:
        t1.writelines([str(instance[0]) + ' ' + str(instance[2]) + '\n' for instance in data_set_michigan])

    with open(args.target_folder_path + '/train2.txt', 'w') as t2:
        t2.writelines([str(instance[1]) + ' ' + str(instance[3]) + '\n' for instance in data_set_michigan])

elif args.process_freiburg:
    summer = []
    winter = []
    data_set_freiburg = []
    processed_season_match = open(folder_path_frei + 'processed_season_match.txt', "r")
    # sorry i realize it might be quite cryptic but i could'nt help myself
    # read lines and set labels to 1 1 (similarity and domain label)
    data_set_freiburg = [array
                        for array in
                        (line.replace('\n',' 1 1').split(' ')
                        for line in processed_season_match.readlines())]
    processed_season_match.close()
    instances_in_line = []
    for instance in data_set_freiburg:
        seasons = []
        i = 1
        while len(instance) - 2 > i:
            seasons = [instance[0], instance[i]]
            random.shuffle(seasons)
            seasons.extend(instance[-2:])
            instances_in_line.append(seasons)
            i += 1
    del data_set_freiburg
    pos_examples = len(instances_in_line)
    neg_examples = 0
    image_gap = 200
    last_index = len(instances_in_line) - 1
    while neg_examples < pos_examples:
        im1 = np.random.randint(0, last_index)
        image_diff = im1 - image_gap - 1
        if image_diff > 0:
            im2 = np.random.randint(0, image_diff)
        else:
            im2 = np.random.randint(im1 + image_gap, last_index)
        seasons = [instances_in_line[im1][0], instances_in_line[im2][1]]
        random.shuffle(seasons)
        seasons.extend([0, 1])
        instances_in_line.append(seasons)
        neg_examples += 1
        if neg_examples % 100 == 0:
            print "{0} / {1}".format(neg_examples, pos_examples)
    random.shuffle(instances_in_line)
    print 'writing txt files'
    freiburg = 'freiburg/'
    with open(args.source_folder_path + '/test1.txt', 'w') as t1:
        t1.writelines([freiburg + str(instance[0]) + ' ' + str(instance[2]) + '\n' for instance in instances_in_line])

    with open(args.source_folder_path + '/test2.txt', 'w') as t2:
        t2.writelines([freiburg + str(instance[1]) + ' ' + str(instance[3]) + '\n' for instance in instances_in_line])
    print "done"
'''
pos_examples = 0
l = 0
for pos_example in data_set_freiburg:
    # we might have one or two winter examples
    if len(pos_example) > 4:
        l += 1
    pos_examples += len(pos_example) - 3
print l
neg_examples = 0
image_gap = 200
last_index = len(data_set_freiburg) - 1
print len(data_set_freiburg)
print "{0} positive examples".format(pos_examples)
print 'creating negative examples'
while neg_examples < pos_examples:
    im1 = np.random.randint(0, last_index)
    image_diff = im1 - image_gap - 1
    if image_diff > 0:
        im2 = np.random.randint(0, image_diff)
    else:
        im2 = np.random.rand(im1 + image_gap, last_index)
    test_instance = data_set_freiburg[im1]
    instance = [test_instance[0]]
    # one winter example
    if len(test_instance) == 4:
        instance.append(test_instance[1])
        neg_examples += 1
    else:
        instance.extend(test_instance[1:2])
        neg_examples += 2
    instance.extend(['0', '1'])
    data_set_freiburg.append(instance)
    if neg_examples % 100 == 0:
        print "{0} / {1}".format(neg_examples, pos_examples)

instances_in_line = []
print 'shuffling and creating final one line instances'
for instance in data_set_freiburg:
    seasons = []
    i = 1
    while len(instance) - 2 > i:
        seasons = [instance[0], instance[i]]
        random.shuffle(seasons)
        seasons.extend(instance[-2:])
        instances_in_line.append(seasons)
        i += 1
print len(data_set_freiburg)
print len(instances_in_line)
del data_set_freiburg
random.shuffle(instances_in_line)
print 'writing txt files'
with open(folder_path_frei + '/train1.txt', 'w') as t1:
    t1.writelines([str(instance[0]) + ' ' + str(instance[2]) + '\n' for instance in instances_in_line])

with open(folder_path_frei + '/train2.txt', 'w') as t2:
    t2.writelines([str(instance[1]) + ' ' + str(instance[3]) + '\n' for instance in instances_in_line])
print "done"
'''
'''
    with open(folder_path_frei + 'processed_season_match.txt', "r") as lines:
        for line_num, line in enumerate(lines):
            line_arr = line.split(' ')
            len_arr = len(line_arr)
            i = 2
            while i + 1 < len_arr:
                summer.append('freiburg/' + line_arr[1] + ' 1')
                winter.append('freiburg/' + line_arr[i + 1] + ' 1')
                i += 2
            if line_num % 100 == 0:
                print str(line_num)
    target_files = {'winter.txt': winter, 'summer.txt': summer}
    for t_file in target_files:
        with open(args.target_folder_path + t_file, 'w') as processed_season_match:
            for line in target_files[t_file]:
                processed_season_match.write("%s\n" % line)
'''
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
