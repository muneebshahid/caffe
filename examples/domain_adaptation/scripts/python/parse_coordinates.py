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
            if sim and curr_ele < threshold:
                closest_args = sorted_args[:j]
                break
            elif not sim and curr_ele >= threshold:
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

    pr_denom = float(true_pos + false_pos)
    recall_denom = float(true_pos + false_neg)
    pr = (true_pos / pr_denom) if pr_denom > 0 else 0
    recall = (true_pos / recall_denom) if recall_denom > 0 else 0
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
    score_data = [#['untrained_places205CNN_iter_300000_upgraded.caffemodel_freiburg_cos_sim.npy',
                   #0.0443317064121, 0.0274826074699, True]]
                  #['mix_data_snapshots_iter_100000.caffemodel_freiburg_cos_sim.npy',
                   #0.037, 0.0213330272421, True],]
                  #['nordland_only_snapshots_iter_100000.caffemodel_freiburg_cos_sim.npy',
                   #0.0372817616013, 0.0265447199265, True]]
                  #['first_four_snapshots_iter_100000.caffemodel_michigan_cos_sim.npy',
                   #0.0223434343434, 0.0198, True]
                    ['mix_data_snapshots_iter_100000.caffemodel_michigan_euc_dist.npy',
                     .0005, 0.02, False]]
    for data in score_data:
        print 'processing: {0}'.format(data[0])
        score_mat = np.load(results_folder + data[0])
        pr_recal_list = []
        #score_mat = np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, score_mat)
        values = np.linspace(data[1], data[2], 100)
        for i, value in enumerate(values):
            pr_recal_result = pr_recall(score_mat, threshold=value, sim=data[-1])
            pr_recal_list.append(pr_recal_result)
            print i, value, pr_recal_result
        np.save(results_folder + 'pr_recall_' + data[0].replace('.npy', ''), np.array(pr_recal_list))
    return


if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    coord_file_path = caffe_root + '/data/domain_adaptation_data/images/coordinates'
    score_txt = caffe_root + '/data/domain_adaptation_data/images/scores.txt'
    results_folder = caffe_root + '/data/domain_adaptation_data/results/'
    main()