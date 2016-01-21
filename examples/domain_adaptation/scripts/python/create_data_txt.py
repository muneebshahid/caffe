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


def split_train_test(data_set, split=0.7):
    train_examples, test_examples = get_train_test_split_len(len(data_set), split)
    # ensures we do not append the same sequence again
    return data_set[0:train_examples], data_set[train_examples:]


def shuffle_columns(instance):
    return [instance[1], instance[0], instance[3], instance[2]]


def get_fukui_im_path(image_id, gt_id, root_folder, is_query=False):
    path = root_folder + ('query/' if is_query else 'db/') + gt_id + image_id
    return path


def extend_data(data):
    temp_data = copy.deepcopy(data)
    random.shuffle(temp_data)
    return temp_data


def get_augmented_data_pos(data_set, ext, limit=4):
    augmented_data = []
    augmented_dict = dict()
    ext_len = len(ext)
    for instance in data_set:
        for im in instance[:2]:
            if im not in augmented_dict:
                augmented_dict[im] = []
        im_1 = instance[0][:-ext_len]
        im_2 = instance[1][:-ext_len]
        keys_dict_1 = {key: [str(rnd) for rnd in random.sample(range(0, 4), limit)] for key in AUGMENTED_KEYS}
        keys_dict_2 = {key: [str(rnd) for rnd in random.sample(range(0, 4), limit)] for key in AUGMENTED_KEYS}

        while len(keys_dict_1) > 0 and len(keys_dict_2) > 0:
            # get actual keys
            key_1, key_2 = random.choice(keys_dict_1.keys()), random.choice(keys_dict_2.keys())

            # pop the elements at the corresponding indices
            aug_im_1, aug_im_2 = keys_dict_1[key_1].pop(), keys_dict_2[key_2].pop()

            # create actual image names
            id_1 = im_1 + '-' + key_1 + aug_im_1 + ext
            id_2 = im_2 + '-' + key_2 + aug_im_2 + ext

            # append to data
            augmented_data.append([instance[0], id_2, instance[2], instance[3]])
            augmented_data.append([instance[1], id_1, instance[2], instance[3]])
            augmented_data.append([id_1, id_2, instance[2], instance[2]])

            # add to dict
            augmented_dict[instance[0]].append(id_1)
            augmented_dict[instance[1]].append(id_2)

            # check if end of dict vars
            if len(keys_dict_1[key_1]) == 0:
                keys_dict_1.pop(key_1)
            if len(keys_dict_2[key_2]) == 0:
                keys_dict_2.pop(key_2)
    return augmented_data, augmented_dict


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


def create_negatives(key, dataset, augmented_length=None, chosen_aug=None, select_orig=0.7):
    negatives = []
    pos_examples = len(dataset) if augmented_length is None else augmented_length
    neg_examples = 0
    fix_dist = False
    image_gap = 0
    if key == 'freiburg':
        image_gap = 200
        print 'creating freiburg negative examples'
    elif key == 'michigan':
        image_gap = 600
        print 'creating michigan negative examples'
    elif key == 'fukui':
        image_gap = 170
        print 'creating fukui negative examples'
    assert image_gap > 0
    while neg_examples < pos_examples:
        im_index_1, im_index_2 = get_distant_images(len(dataset), image_gap, fix_dist)
        ims = dataset[im_index_1], dataset[im_index_2]
        if augmented_length is None:
            negatives.append([ims[0][0], ims[1][1], 0, 0])
        else:
            instance = []
            im_file = []
            im_file.extend(osh.split_file_extension(ims[0][0]))
            im_file.extend(osh.split_file_extension(ims[1][1]))
            likelihoods = np.random.rand(2, 1)
            for i, likelihood in enumerate(likelihoods):
                if likelihood < select_orig:
                    instance.append(ims[i][i])
                elif chosen_aug is not None:
                        instance.append(random.choice(chosen_aug[ims[i][i]]))
                else:
                    f_name, f_ext = im_file[i]
                    aug_key = random.choice(AUGMENTED_KEYS)
                    num = random.choice(range(0, 3))
                    full_im = f_name + '-' + aug_key + str(num) + f_ext
                    instance.append(full_im)
            instance.extend([0, 0])
            negatives.append(instance)
        if neg_examples % 1000 == 0:
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
                                 (line.replace('summer/', 'freiburg/summer')
                                      .replace('winter/', 'freiburg/winter')
                                      .replace('\n', ' 1 1').split(' ')
                                  for line in data_reader.readlines())]
            for instance in data_set_freiburg:
                i = 1
                while len(instance) - 2 > i:
                    seasons = [instance[0], instance[i]]
                    seasons.extend(instance[-2:])
                    data_set.append(seasons)
                    i += 1
    elif key == 'michigan':
        mich_ignore = range(1264, 1272)
        mich_ignore.extend(range(1473, 1524))
        mich_ignore.extend(range(1553, 1565))
        mich_ignore.extend(range(1623, 1628))
        # indoor
        mich_ignore.extend(range(4795, 5521))
        mich_ignore.extend(range(8324, 9288))
        mich_ignore.extend(range(10095, 10677))
        mich_ignore.extend(range(11270, 11776))
        mich_ignore.extend(range(11985, 12575))
        folder_path = root_folder_path + 'michigan/'
        print 'Processing michigan data.....'
        files_michigan = ['michigan/aug/' + im + '.tiff'
                          for im in
                          sorted([im[:-5] for im in
                                  osh.list_dir(folder_path + 'aug/')], key=int)]

        for im_file in files_michigan:
            file_n = osh.extract_name_from_path(im_file)
            if int(file_n[:-5]) in mich_ignore:
                print "ignoring {0}".format(file_n[:-5])
                continue
            data_set.append([im_file, im_file.replace('aug', 'jan'), 1, 1])
    elif key == 'fukui':
        # mislabeled examples
        fukui_ignore = {'SU': ['4', '04000000'],
                        'SP': ['4', '04000000', '04000001', '04000002'],
                        'AU': ['1', '01000000', '01000001', '01000002', '01000003'],
                        'WI': ['None']}
        folder_path = root_folder_path + 'fukui/'
        print 'processing fukui data.....'
        extra_im_range = 3
        seasons = ['AU', 'SP', 'SU', 'WI']
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
                        # skip mislabeled examples
                        if gt == fukui_ignore[season][0] and qu_image in fukui_ignore[season]:
                            print 'ignoring {0}, gt {1}, season {2}'.format(qu_image, gt, season)
                            continue
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
                                      qu_image_path.replace(root_folder_path, ''), 1, 1]

                        data_set.append(gt_example)
                        int_db_image = int(db_image)
                        im_range = range(int(images[2]), int(images[3]) + 1)

                        # append images before if possible
                        for im in range(int_db_image - extra_im_range, int_db_image):
                            if im in im_range:
                                db_image_path = get_fukui_im_path('0' + str(im) + '.jpg', gt + '/', season_folder)
                                if osh.is_file(db_image_path):
                                    gt_example = [db_image_path.replace(root_folder_path, ''),
                                                  qu_image_path.replace(root_folder_path, ''), 1, 1]
                                    data_set.append(gt_example)

                        # append images after if possible
                        for im in range(int_db_image + 1, int_db_image + extra_im_range + 1):
                            if im in im_range:
                                db_image_path = get_fukui_im_path('0' + str(im) + '.jpg', gt + '/', season_folder)
                                if osh.is_file(db_image_path):
                                    gt_example = [db_image_path.replace(root_folder_path, ''),
                                                  qu_image_path.replace(root_folder_path, ''), 1, 1]
                                    data_set.append(gt_example)
    return data_set


def write_data(data_set, root_folder_path, write_path, file_path, file_num=None, lmdb=True):
    file_path = write_path + '/' + file_path
    if file_num is not None:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
            w.writelines(
                    [('' if lmdb else root_folder_path) + str(instance[file_num - 1]).replace('\\', '/') +
                     ' ' + str(instance[file_num + 1]) + '\n' for instance in data_set])
    else:
        with open(file_path + '.txt', 'w') as w:
            # file1 uses columns 0 and 2, while file2 uses columns 1 and 3
           w.writelines(
                    [str(instance[0]).replace('\\', '/') + ' ' + str(instance[-1]) + '\n' for instance
                     in data_set])


def evenly_mix_pos_neg(data_pos, data_neg, batch_size=8):
    data = []
    i = 0
    j = 0
    # assumes neg >= pos
    end_len = len(data_neg)
    batch_half = batch_size / 2
    while True:
        if i <= len(data_neg) - batch_half:
            batch_neg = data_neg[i:i + batch_half]
            i += batch_half
        else:
            batch_neg = data_neg[i:]
            break
        if i <= len(data_pos) - batch_half:
            batch_pos = data_pos[i:i + batch_half]
            j += batch_half
        else:
            batch_pos = data_pos[i:]
            j = 0
        batch = batch_pos
        batch.extend(batch_neg)
        random.shuffle(batch)
        data.extend(batch)
    return data


def pseudo_shuffle_data(data, pseudo_shuffle):
    if pseudo_shuffle > 0:
        data_orig = copy.deepcopy(data)
        i = 0
        while i < pseudo_shuffle:
            print "extending train data {0} time".format(i + 1)
            data.extend(extend_data(data_orig))
            i += 1


def process_datasets(keys, root_folder_path, write_path, pseudo_shuffle=0, augmented_limit=1):
    train_data_pos = []
    train_data_neg = []
    test_data_pos = []
    test_data_neg = []
    for key in keys:
        ext = '.jpg' if key != 'michigan' else '.tiff'
        data_set_pos = get_dataset(key, root_folder_path)
        data_set_neg = create_negatives(key, data_set_pos)
        # Add fukui data only for training.
        if key != 'fukui':
            train_data_pos_temp, test_data_pos_temp = split_train_test(data_set_pos)
            train_data_neg_temp, test_data_neg_temp = split_train_test(data_set_neg)
            train_data_pos.extend(train_data_pos_temp)
            train_data_neg.extend(train_data_neg_temp)
            test_data_pos.extend(test_data_pos_temp)
            test_data_neg.extend(test_data_neg_temp)
        else:
            train_data_pos_temp, train_data_neg_temp = data_set_pos, data_set_neg
            train_data_pos.extend(train_data_pos_temp)
            train_data_neg.extend(train_data_neg_temp)

        augmented_pos, augmented_dict = get_augmented_data_pos(train_data_pos_temp, ext, augmented_limit)
        augmented_neg = create_negatives(key, train_data_pos_temp, len(augmented_pos), augmented_dict, 0.7)
        train_data_pos.extend(augmented_pos)
        train_data_neg.extend(augmented_neg)

    random.shuffle(train_data_pos)
    random.shuffle(train_data_neg)

    print "train data pos {0}, train data neg {1}".format(len(train_data_pos), len(train_data_neg))
    pseudo_shuffle_data(train_data_pos, pseudo_shuffle)
    pseudo_shuffle_data(train_data_neg, pseudo_shuffle)
    print "extended train data pos {0}, train data neg {1}".format(len(train_data_pos), len(train_data_neg))

    train_data = evenly_mix_pos_neg(train_data_pos, train_data_neg, 8)
    test_data = test_data_pos
    test_data.extend(test_data_neg)

    print "train data {0}".format(len(train_data))
    print "test data {0}".format(len(test_data))

    print 'writing files....'
    write_data(train_data, root_folder_path, write_path, 'train1', file_num=1, lmdb=False)
    write_data(train_data, root_folder_path, write_path, 'train2', file_num=2, lmdb=False)
    write_data(test_data, root_folder_path, write_path, 'test1', file_num=1, lmdb=False)
    write_data(test_data, root_folder_path, write_path, 'test2', file_num=2, lmdb=False)


def main():
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    root_folder_path = caffe_root + '/../data/images/orig/'
    root_folder_path = root_folder_path.replace('\\', '/')
    if not osh.is_dir(root_folder_path):
        print "source folder does'nt exist, existing....."
        sys.exit()
    keys = ['freiburg', 'michigan', 'fukui']
    write_path = caffe_root + '/data/domain_adaptation_data/images/'
    process_datasets(keys, root_folder_path, write_path)

if __name__ == "__main__":
    AUGMENTED_KEYS = ['tra' , 'rot', 'aff', 'per']
    main()
