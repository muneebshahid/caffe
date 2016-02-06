import os_helper as osh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_sorted_arg(row, sim):
    if sim:
        return np.argsort(row)[::-1]
    else:
        return np.argsort(row)

def create_score_mat(qu, db):
    score_mat = np.ones((qu.shape[0], qu.shape[0]))
    for i, qu_point in enumerate(qu):
        for j, db_point in enumerate(db):
            score_mat[i, j] = np.linalg.norm(qu_point - db_point)
        if i % 50 == 0:
            print i
    return np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, score_mat)


def pr_recall(score_mat, sim=True, im_range=3, threshold=.045):
    count = 0
    true_pos, false_neg, false_pos = 0, 0, 0
    for i, row in enumerate(score_mat):
        true_pos_found = False
        sorted_args = get_sorted_arg(row, sim)
        closest_args = []
        for j, args in enumerate(sorted_args):
            curr_ele = score_mat[i, args]
            if curr_ele < threshold:
                closest_args = sorted_args[:j]
                break

        range_arr = range(i - im_range, i + im_range + 1)
        for args in closest_args:
            if args in range_arr:
                true_pos_found = True
            else:
                false_pos += 1
        if true_pos_found:
            true_pos += 1
        else:
            false_neg += 1
    pr = true_pos / float(true_pos + false_pos)
    recall = true_pos / float(true_pos + false_neg)
    return pr, recall


def vals_around_diag(score_mat, sim=True, k=3, diag=5):
    values_inside, values_outside = 0, 0
    for i, row in enumerate(score_mat):
        sorted_args = get_sorted_arg(row,sim)[:k]
        range_arr = range(i - diag, i + diag +1)
        for min_arg in sorted_args:
            if min_arg in range_arr:
                values_inside += 1
            else:
                values_outside += 1
    total_pts = values_inside + values_outside
    return values_inside / float(total_pts)


def main():
    #freiburg_coords = np.load(coord_file_path + '0.npy')
    score_mats = ['scores_untrained.npy', 'scores_iter_20k.npy', 'scores_iter_60k.npy']
    for score_mat in score_mats:
        score_mat = np.load(features_folder + score_mat)
        #np.save(features_folder + score_file, score_mat)
        score_mat = np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, score_mat)
        print pr_recall(score_mat, threshold=.035)
        #print vals_around_diag(score_mat)
    return


if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    coord_file_path = caffe_root + '/data/domain_adaptation_data/images/coordinates'
    score_txt = caffe_root + '/data/domain_adaptation_data/images/scores.txt'
    features_folder = caffe_root + '/data/domain_adaptation_data/features/'
    main()