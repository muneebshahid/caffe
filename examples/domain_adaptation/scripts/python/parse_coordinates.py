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


def accuracy(score_mat):
    count = 0
    for i, row in enumerate(score_mat):
        count += (1 if i in np.argsort(row)[:5] else 0)
    return count


def main():
    freiburg, michigan = read_files()
    freiburg_qu, freiburg_db = freiburg[:, :3], freiburg[:, 3:]
    michigan_qu, michigan_db = michigan[:, :3], michigan[:, 3:]
    #plot_data(freiburg_qu[:100], freiburg_db[:100])score
    score_mat = create_score_mat(freiburg_qu, freiburg_db)
    #score_mat = np.loadtxt(score_txt)
    np.savetxt(caffe_root + '/data/domain_adaptation_data/images/scores.txt', score_mat, '%10.5f')
    print accuracy(score_mat)
    return


if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    coord_file_txt = caffe_root + '/data/domain_adaptation_data/images/coordinates'
    score_txt = caffe_root + '/data/domain_adaptation_data/images/scores.txt'
    main()