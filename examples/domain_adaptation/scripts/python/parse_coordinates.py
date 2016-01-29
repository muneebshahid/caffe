import os_helper as osh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_files():
    return np.loadtxt(coord_file_txt + '0.txt'), np.loadtxt(coord_file_txt + '1.txt')


def plot_data(qu, db):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qu[:, 0], qu[:, 1], qu[:, 2], c='r')
    ax.scatter(db[:, 0], db[:, 1], db[:, 2], c='b')
    for qu_point, db_point in zip(qu, db):
        ax.plot([qu_point[0], db_point[0]], [qu_point[1], db_point[1]], [qu_point[2], db_point[2]], c='g')
    plt.show()


def create_score_mat(qu, db):
    score_mat = np.ones((qu.shape[0], qu.shape[0]))
    for i, qu_point in enumerate(qu):
        for j, db_point in enumerate(db):
            score_mat[i, j] = np.linalg.norm(qu_point - db_point)
        if i % 50 == 0:
            print i
    return score_mat


def pr_recall(score_mat, im_range=3, threshold=.35):
    count = 0
    true_pos, false_neg, false_pos = 0, 0, 0
    for i, row in enumerate(score_mat):
        true_pos_found = False
        sorted_args = np.argsort(row)
        closest_args = []
        for j, args in enumerate(sorted_args):
            curr_ele = score_mat[i, args]
            if curr_ele > threshold:
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


def main():
    freiburg, michigan = read_files()
    print freiburg.shape
    freiburg_qu, freiburg_db = freiburg[:, :128], freiburg[:, 128:]
    print freiburg_qu.shape, freiburg_db.shape
    michigan_qu, michigan_db = michigan[:, :3], michigan[:, 3:]
    #plot_data(freiburg_qu[:100], freiburg_db[:100])score
    #score_mat = create_score_mat(freiburg_qu, freiburg_db)
    score_mat = np.loadtxt(score_txt)
    np.savetxt(caffe_root + '/data/domain_adaptation_data/images/scores.txt', score_mat, '%10.5f')
    print pr_recall(score_mat)
    return


if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    coord_file_txt = caffe_root + '/data/domain_adaptation_data/images/coordinates'
    score_txt = caffe_root + '/data/domain_adaptation_data/images/scores.txt'
    main()