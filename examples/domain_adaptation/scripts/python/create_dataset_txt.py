import os_helper as osh
import sys
import numpy as np
import random
import cv2
import copy
from itertools import izip_longest


mich_ignore = range(1264, 1272)
mich_ignore.extend(range(1473, 1524))
mich_ignore.extend(range(1553, 1565))
mich_ignore.extend(range(1623, 1628))


def get_train_test_split_len(examples_len, split):
    # put first split% of the data for training and the rest for testing
    return int(np.ceil(split * examples_len)), -int(np.floor((1 - split) * examples_len))


def split_source_target(data_set, sources, targets, label_data_limit):
    if len(sources) == 0 or len(targets) == 0:
        print "Source or target cannot be empty"
        sys.exit()
    data_set_source = []
    data_set_target = []
    data_set_test = []
    for source in sources:
        pos_source_data = data_set[source][0]
        neg_source_data = data_set[source][1]

        data_set_source.extend(data_set[source][0])
        data_set_source.extend(data_set[source][1])

    for target in targets:
        # negative examples
        data_set_target.extend(data_set[target][1])
        data_set_test.extend(data_set[target][1])

        label_data_indices = []
        pos_target_data = copy.deepcopy(data_set[target][0])
        # semi supervised
        if label_data_limit > 0:
            while len(label_data_indices) < label_data_limit:
                index = np.random.randint(0, len(pos_target_data) - 1)
                if index in label_data_indices:
                    continue
                label_data_indices.append(index)
                data_set_source.append(pos_target_data[index])

            for i, instance in enumerate(pos_target_data):
                if i in label_data_indices:
                    continue
                data_set_test.append(instance)
        # unsupervised
        else:
            data_set_test.extend(pos_target_data)

        while len(pos_target_data) > 1:
            index = np.random.randint(0, len(pos_target_data) - 1)
            pos_ins_1 = pos_target_data.pop(index)
            if len(pos_target_data) - 1 > 0:
                index = np.random.randint(0, len(pos_target_data) - 1)
            else:
                index = 0
            pos_ins_2 = pos_target_data.pop(index)
            pos_ins_3 = [pos_ins_1[0], pos_ins_2[1], pos_ins_1[2], pos_ins_1[3]]
            pos_ins_4 = [pos_ins_2[0], pos_ins_1[1], pos_ins_1[2], pos_ins_1[3]]
            # since we do not
            data_set_target.extend([pos_ins_3, pos_ins_4])
    # shuffling data
    random.shuffle(data_set_source)
    random.shuffle(data_set_target)
    random.shuffle(data_set_test)
    return data_set_source, data_set_target, data_set_test


def get_batch(index, batch_size, data):
    len_data = len(data)
    reached_end = False
    if index + batch_size < len_data:
        batch = data[index: index + batch_size]
        index += batch_size
    else:
        batch = data[index:]
        # randint: len_source - 1
        rand_indices = np.random.randint(0, len_data, batch_size - len(batch))
        batch.extend([data[index] for index in rand_indices])
        reached_end = True
    return batch, reached_end


def pad_source_target(source, target, batch_size=320):
    half_batch_size = batch_size / 2
    len_source = len(source)
    len_target = len(target)
    print "source len: {0}, target len {1}, batch {2}".format(len_source, len_target, batch_size)
    # half of the data set in each batch should be test data
    padded = []
    end = False
    s_index = 0
    t_index = 0

    while not end:
        s_batch, s_reached_end = get_batch(s_index, half_batch_size, source)
        assert len(s_batch) == half_batch_size
        t_batch, t_reached_end = get_batch(t_index, half_batch_size, target)
        assert len(t_batch) == half_batch_size
        padded.extend(s_batch)
        padded.extend(t_batch)

        if t_reached_end:
            t_index = 0
            random.shuffle(target)
        else:
            t_index += half_batch_size
        s_index += half_batch_size
        end = s_reached_end
    assert len(padded) % batch_size == 0
    return padded


def split_data(train, test, pos, neg, split):
    pos_train_examples, pos_test_examples = get_train_test_split_len(len(pos), split)
    neg_train_examples, neg_test_examples = get_train_test_split_len(len(neg), split)
    # ensures we do not append the same sequence again
    random.shuffle(pos)
    random.shuffle(neg)
    for pos_example, neg_example in izip_longest(pos[0:pos_train_examples], neg[0:neg_train_examples]):
        if pos_example is not None:
            train.append(pos_example)
        if neg_example is not None:
            train.append(neg_example)
    for pos_example, neg_example in izip_longest(pos[pos_test_examples:], neg[neg_test_examples:]):
        if pos_example is not None:
            test.append(pos_example)
        if neg_example is not None:
            test.append(neg_example)
    # ensures an already processed data set does not always stay at the beginning
    random.shuffle(train)
    random.shuffle(test)


def progress(current, total):
    return '{0} / {1}'.format(current, total)


def write(data_set, file_path, file_num = None):
    if file_num is not None:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
            w.writelines(
                    [str(instance[file_num - 1]).replace('\\', '/') + ' ' + str(instance[file_num + 1]) + '\n' for instance
                     in data_set])
    else:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
            w.writelines([instance for instance in data_set])


def process_freiburg(data_set, source, key, folder_path):
    print "processing freiburg data....."
    save_neg_im = False
    data_set_freiburg_pos = []
    data_set_freiburg_neg = []
    processed_season_match = open(folder_path + 'processed_season_match.txt', "r")
    # sorry i realize it might be quite cryptic but i could'nt help myself
    # read lines and set labels to 1 1 (similarity and domain label)
    data_set_freiburg = [array
                         for array in
                         (line.replace('uncompressed', 'freiburg/uncompressed')
                              .replace('\n', ' 1 ' + str(int(source))).split(' ')
                                for line in processed_season_match.readlines())]
    processed_season_match.close()
    for instance in data_set_freiburg:
        i = 1
        while len(instance) - 2 > i:
            seasons = [instance[0], instance[i]]
            random.shuffle(seasons)
            seasons.extend(instance[-2:])
            data_set_freiburg_pos.append(seasons)
            i += 1
    del data_set_freiburg
    pos_examples = len(data_set_freiburg_pos)
    neg_examples = 0
    image_gap = 200
    last_index = len(data_set_freiburg_pos) - 1
    while neg_examples < pos_examples:
        im1 = np.random.randint(0, last_index)
        image_diff = im1 - image_gap - 1
        if image_diff > 0:
            im2 = np.random.randint(0, image_diff)
        else:
            im2 = np.random.randint(im1 + image_gap, last_index)
        if save_neg_im:
            im_1 = cv2.imread(folder_path + data_set_freiburg_pos[im1][0])
            im_2 = cv2.imread(folder_path + data_set_freiburg_pos[im2][1])
            cv2.imwrite(folder_path + 'neg_im/' + str(neg_examples) + '.png',
                        np.concatenate((im_1, im_2), axis=1))
            print 'saving neg example {0} / {1}'.format(neg_examples, pos_examples)
        seasons = [data_set_freiburg_pos[im1][0], data_set_freiburg_pos[im2][1]]
        random.shuffle(seasons)
        seasons.extend([0, int(source)])
        data_set_freiburg_neg.append(seasons)
        if neg_examples % 100 == 0:
            print "{0} / {1}".format(neg_examples, pos_examples)
        neg_examples += 1
    data_set[key] = [data_set_freiburg_pos, data_set_freiburg_neg]


def process_michigan(data_set, source, key, folder_path):
        print 'Processing michigan....'
        data_set_michigan = []
        data_set_michigan_pos = []
        data_set_michigan_neg = []
        # Process Michigan data
        months_mich = ['aug', 'jan']
        files_michigan_pos = [im.replace('\\', '/')
                              for im in sorted(osh.get_folder_contents(folder_path + 'jan/', '*.tiff'))]

        print 'Creating positive examples'
        for jan_file in files_michigan_pos:
            file_n = osh.extract_name_from_path(jan_file)

            if int(file_n[5:-5]) in mich_ignore:
                print "ignoring {0}".format(file_n[5:-5])
                continue
            jan_file = jan_file.replace(root_folder_path, '')
            month_mich = np.random.random_integers(0, 1)
            path_1 = jan_file.replace('jan', months_mich[month_mich])
            path_2 = jan_file.replace('jan', months_mich[abs(month_mich - 1)])
            # label_1 is similarity label, while label_2 is the domain_label

            data_set_michigan_pos.append([path_1, path_2, 1, int(source)])

        del files_michigan_pos
        images_gap = 600
        michigan_neg_instances = 0
        michigan_pos_len = len(data_set_michigan_pos)
        last_index = michigan_pos_len - 1
        print "found {0} positive examples".format(michigan_pos_len)
        print 'Creating negative examples'
        while michigan_neg_instances < michigan_pos_len:
            im1 = np.random.random_integers(0, last_index)
            month_mich = np.random.random_integers(0, 1)
            file_1 = folder_path + months_mich[month_mich] + '/00000' + str(im1) + '.tiff'
            # michigan data set has a weird naming convention, so checking if the file actually exists
            if not osh.is_file(file_1):
                continue
            # ensure image gap between negative examples
            im_diff = im1 - images_gap
            if im_diff > 0:
                im2 = np.random.random_integers(0, im_diff)
            else:
                im2 = np.random.randint(im1 + images_gap, last_index)
            file_2 = folder_path + months_mich[abs(month_mich - 1)] + '/00000' + str(im2) + '.tiff'
            if not osh.is_file(file_2):
                continue
            file_1 = file_1.replace(folder_path, 'michigan/uncompressed/')
            file_2 = file_2.replace(folder_path, 'michigan/uncompressed/')
            data_set_michigan_neg.append([file_1, file_2, 0, int(source_mich)])
            michigan_neg_instances += 1
            if michigan_neg_instances % 2000 == 0:
                print 'negative examples: ', progress(michigan_neg_instances, michigan_pos_len)

        print "created {0} negative examples".format(len(data_set_michigan_neg))
        data_set[key] = [data_set_michigan_pos, data_set_michigan_neg]


def main(label_data_limit=0):
    source = []
    target = []
    data_set = {}
    if source_mich is not None:
        key = 'michigan'
        if source_mich:
            source.append(key)
        else:
            target.append(key)
        folder_path = root_folder_path + 'michigan/uncompressed/'
        process_michigan(data_set, source_mich, key, folder_path)

    if source_freiburg is not None:
        key = 'freiburg'
        if source_freiburg:
            source.append(key)
        else:
            target.append(key)
        folder_path = root_folder_path + key + '/'
        process_freiburg(data_set, source_freiburg, key, folder_path)

    print "splitting in to target and source"
    source_data, target_data, test_data = split_source_target(data_set, source, target, label_data_limit)

    print "padding source data with target data"
    padded = pad_source_target(copy.deepcopy(source_data), copy.deepcopy(target_data), 256)

    print "writing data files"
    write(padded, root_folder_path + 'train1', 1)
    write(padded, root_folder_path + 'train2', 2)
    write(source_data, root_folder_path + 'source1', 1)
    write(source_data, root_folder_path + 'source2', 2)
    write(target_data, root_folder_path + 'target1', 1)
    write(target_data, root_folder_path + 'target2', 2)
    write(test_data, root_folder_path + 'test1', 1)
    write(test_data, root_folder_path + 'test2', 2)

    source_target_data_set = []
    print "creating data set for image mean"
    for domain in data_set:
        instances = data_set[domain]
        for pos, neg in izip_longest(instances[0], instances[1]):
            pos_neg = [pos, neg]
            for instance in pos_neg:
                if instance is not None:
                    for im in instance[:-2]:
                        im += ' 1\n'
                        if im not in source_target_data_set:
                            source_target_data_set.append(im)
    print "writing data set for image mean"
    write(source_target_data_set, root_folder_path + 'complete')

if __name__ == "__main__":
    root_folder_path = osh.get_env_var('CAFFE_ROOT') + '/../data/images/' + '/'
    root_folder_path = root_folder_path.replace('\\', '/')
    if not osh.is_dir(root_folder_path):
        print "source folder does'nt exist, existing....."
        sys.exit()
    pad = False
    source_mich = True
    source_freiburg = False
    main()
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
