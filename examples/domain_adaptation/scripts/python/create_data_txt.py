import os_helper as osh
import sys
import numpy as np
import random
import cv2
import copy
from itertools import izip_longest


def get_train_test_split_len(examples_len, split):
    # put first split% of the data for training and the rest for testing
    return int(np.ceil(split * examples_len)), -int(np.floor((1 - split) * examples_len))


def split_train_test(data_set, split=0.8):
    random.shuffle(data_set)
    train_examples, test_examples = get_train_test_split_len(len(data_set), split)
    # ensures we do not append the same sequence again
    return data_set[0:train_examples], data_set[train_examples:]


def shuffle_columns(instance):
    return [instance[1], instance[0], instance[3], instance[2], instance[-1]]


def get_fukui_im_path(image_id, gt_id, root_folder, is_query=False):
    path = root_folder + ('query/' if is_query else 'db/') + gt_id + image_id
    return path


def get_distant_images(data_len, image_gap, fix_dist=False):
    im_index_1 = np.random.randint(0, data_len)
    if not fix_dist:
        im_diff = im_index_1 - image_gap - 1
        while True:
            im_index_2 = np.random.randint(0, data_len)
            if abs(im_index_1 - im_index_2) > im_diff:
                break
    else:
        if im_index_1 + image_gap >= data_len:
            im_index_2 = im_index_1 - image_gap
        else:
            im_index_2 = im_index_1 + image_gap
    return im_index_1, im_index_2

def evenly_mix_source_target(dataset, batch_size=8):
    i = 0
    while True:
        if i <= len(dataset) - batch_size:
            batch = dataset[i:i + batch_size]
            for j, instance in enumerate(batch[0:len(batch)/2]):
                batch[j] = shuffle_columns(instance)
            random.shuffle(batch)
            dataset[i:i + batch_size] = batch
        else:
            batch = dataset[i:]
            for j, instance in enumerate(batch[0:len(batch)/2]):
                batch[j] = shuffle_columns(instance)
            random.shuffle(batch)
            dataset[i:] = batch
            break
        i += batch_size


def create_negatives(key, dataset):
    negatives = []
    if key == 'freiburg':
        print 'creating freiburg negative examples'
        pos_examples = len(dataset)
        neg_examples = 0
        image_gap = 200
        while neg_examples < pos_examples:
            im_index_1, im_index_2 = get_distant_images(len(dataset), image_gap)
            im1 = dataset[im_index_1]
            im2 = dataset[im_index_2]
            negatives.append([im1[0], im2[1], im1[2], im2[3], 0])
            if neg_examples % 100 == 0:
                print "{0} / {1}".format(neg_examples, pos_examples)
            neg_examples += 1
    elif key == 'michigan':
        print 'creating michigan negative examples'
        pos_examples = len(dataset)
        neg_examples = 0
        image_gap = 600
        while neg_examples < pos_examples:
            im_index_1, im_index_2 = get_distant_images(len(dataset), image_gap)
            im1 = dataset[im_index_1]
            im2 = dataset[im_index_2]
            negatives.append([im1[0], im2[1], im1[2], im2[3], 0])
            if neg_examples % 100 == 0:
                print "{0} / {1}".format(neg_examples, pos_examples)
            neg_examples += 1
    elif key == 'fukui':
        print 'creating fukui negative examples'
        pos_examples = len(dataset)
        neg_examples = 0
        image_gap = 170
        while neg_examples < pos_examples:
            im_index_1, im_index_2 = get_distant_images(len(dataset), image_gap, True)
            im1 = dataset[im_index_1]
            im2 = dataset[im_index_2]
            negatives.append([im1[0], im2[1], im1[2], im2[3], 0])
            if neg_examples % 100 == 0:
                print "{0} / {1}".format(neg_examples, pos_examples)
            neg_examples += 1
    return negatives


def get_dataset(key, root_folder_path):
    data_set = []
    if key == 'freiburg':
        folder_path = root_folder_path + 'freiburg/'
        print "processing freiburg data....."
        save_neg_im = False
        data_set_freiburg_pos = []
        data_set_freiburg_neg = []
        with open(folder_path + 'processed_season_match.txt', "r") as data_reader:
            # sorry i realize it might be quite cryptic but i could'nt help myself
            # read lines and set labels to 1 1 (similarity and domain label)
            data_set_freiburg = [array
                                 for array in
                                 (line.replace('uncompressed', 'freiburg/uncompressed')
                                      .replace('\n', ' 1 0 1').split(' ')
                                        for line in data_reader.readlines())]
            for instance in data_set_freiburg:
                i = 1
                while len(instance) - 3 > i:
                    seasons = [instance[0], instance[i]]
                    seasons.extend(instance[-3:])
                    data_set.append(seasons)
                    i += 1
    elif key == 'michigan':
        mich_ignore = range(1264, 1272)
        mich_ignore.extend(range(1473, 1524))
        mich_ignore.extend(range(1553, 1565))
        mich_ignore.extend(range(1623, 1628))
        folder_path = root_folder_path + 'michigan/uncompressed/'
        print 'Processing michigan data.....'
        files_michigan = ['michigan/uncompressed/aug/' + im + '.tiff'
                          for im in
                          sorted([im[:-5] for im in
                                  osh.list_dir(folder_path + 'aug/')], key=int)]

        for im_file in files_michigan:
            file_n = osh.extract_name_from_path(im_file)
            if int(file_n[:-5]) in mich_ignore:
                print "ignoring {0}".format(file_n[5:-5])
                continue
            data_set.append([im_file, im_file.replace('aug', 'jan'), 1, 0, 1])
    elif key == 'fukui':
        folder_path = root_folder_path + 'fukui/'
        print 'processing fukui data.....'
        extra_im_range = 3
        seasons = ['WI']
        for season in seasons:
            print 'processing: ', season
            season_folder = folder_path + season + '/'
            ground_truth_folder = season_folder + 'gt/'
            gts = sorted([gt[:-4] for gt in osh.list_dir(ground_truth_folder)], key=int)
            print 'creating positive examples'
            for gt in gts:
                with open(ground_truth_folder + gt + '.txt', "r") as ground_truths:
                    for line in ground_truths.readlines():
                        # skip \n
                        images = line.replace('\n', '').split(' ')
                        qu_image = images[0]
                        db_image = images[1]

                        qu_image_path = get_fukui_im_path(qu_image + '.jpg', gt + '/', season_folder, True)
                        db_image_path = get_fukui_im_path(db_image + '.jpg', gt + '/', season_folder)
                        if not osh.is_file(qu_image_path):
                            print "query not found", qu_image_path
                        if not osh.is_file(db_image_path):
                            print '---------------'
                            print 'db not found'
                            print qu_image_path
                            print db_image_path
                            print '---------------'
                        if int(db_image) < int(images[2]):
                            print 'db less than limit'
                            print qu_image_path
                            print db_image_path
                            print '----------------'
                        gt_example = [db_image_path.replace(root_folder_path, ''),
                                      qu_image_path.replace(root_folder_path, ''), 1, 0, 1]

                        data_set.append(gt_example)
                        int_db_image = int(db_image)
                        im_range = range(int(images[2]), int(images[3]) + 1)

                        # append images before if possible
                        for im in range(int_db_image - extra_im_range, int_db_image):
                            if im in im_range:
                                db_image_path = get_fukui_im_path('0' + str(im) + '.jpg', gt + '/', season_folder)
                                if osh.is_file(db_image_path):
                                    gt_example = [db_image_path.replace(root_folder_path, ''),
                                                  qu_image_path.replace(root_folder_path, ''), 1, 0, 1]
                                    data_set.append(gt_example)

                        # append images after if possible
                        for im in range(int_db_image + 1, int_db_image + extra_im_range + 1):
                            if im in im_range:
                                db_image_path = get_fukui_im_path('0' + str(im) + '.jpg', gt + '/', season_folder)
                                if osh.is_file(db_image_path):
                                    gt_example = [db_image_path.replace(root_folder_path, ''),
                                                  qu_image_path.replace(root_folder_path, ''), 1, 0, 1]
                                    data_set.append(gt_example)
    return data_set


def write_data(data_set, file_path, file_num = None):
    if file_num is not None:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
            w.writelines(
                    [str(instance[file_num - 1]).replace('\\', '/') + ' ' + str(instance[file_num + 1]) + '\n' for instance
                     in data_set])
    else:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
           w.writelines(
                    [str(instance[0]).replace('\\', '/') + ' ' + str(instance[-1]) + '\n' for instance
                     in data_set])


def process_datasets(keys, root_folder_path):
    train_data = []
    test_data = []
    for key in keys:
        data_set_pos = get_dataset(key, root_folder_path)
        data_set_neg = create_negatives(key, data_set_pos)
        train_data_pos_temp, test_data_pos_temp = split_train_test(data_set_pos)
        train_data_neg_temp, test_data_neg_temp = split_train_test(data_set_neg)
        train_data.extend(train_data_pos_temp)
        train_data.extend(train_data_neg_temp)
        test_data.extend(test_data_pos_temp)
        test_data.extend(test_data_neg_temp)

    random.shuffle(train_data)
    random.shuffle(test_data)

    #evenly_mix_source_target(train_data)
    #evenly_mix_source_target(test_data)

    write_data(train_data, root_folder_path + 'train')
    write_data(train_data, root_folder_path + 'train1', 1)
    write_data(train_data, root_folder_path + 'train2', 2)
    write_data(test_data, root_folder_path + 'test')
    write_data(test_data, root_folder_path + 'test1', 1)
    write_data(test_data, root_folder_path + 'test2', 2)


def main(label_data_limit=0):
    root_folder_path = osh.get_env_var('CAFFE_ROOT') + '/../data/images/' + '/'
    root_folder_path = root_folder_path.replace('\\', '/')
    # batch size is used for padding.
    batch_size = 128

    # flag to pad source and target arrays to make them a multiple of batch size
    pad_multiple = True

    # flag to pad train data (source) with target
    pad_train = True

    create_mean_data = False
    # until a custom shuffling is implementd in the data layer,
    # pseudo shuffle the data by extending it with random repititons
    # of the whole data set
    pseudo_shuffle = 5
    source_mich = True
    source_freiburg = False
    source_fukui = True
    source = []
    target = []
    source_data = []
    target_data = []

    if not osh.is_dir(root_folder_path):
        print "source folder does'nt exist, existing....."
        sys.exit()
    keys = ['freiburg', 'michigan', 'fukui']
    process_datasets(keys, root_folder_path)

if __name__ == "__main__":
    main()