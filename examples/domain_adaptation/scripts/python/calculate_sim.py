import numpy as np
import os_helper as osh


def dump_results(results):
    np.save(result_path + save_file, results)


def normalize(feature):
    flattened_features = feature.flatten().astype(dtype=np.float32)
    return flattened_features / np.linalg.norm(flattened_features)


def normalize_matrix_rows(matrix):
    return np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, matrix)


def create_score_mat(qu, db):
    score_mat = np.ones((qu.shape[0], qu.shape[0]))
    for i, qu_point in enumerate(qu):
        for j, db_point in enumerate(db):
            score_mat[i, j] = np.linalg.norm(qu_point - db_point)
        if i % 100 == 0:
            print i
    return score_mat


def main(sim, f1, f2, normalize_features=True, normalize_matrix=True):
    score_mat = None
    if sim == 'euc':
        score_mat = create_score_mat(f1, f2)
        if normalize_matrix:
            score_mat = normalize_matrix_rows(score_mat)
    elif sim == 'cos':
        if normalize_features:
            f1, f2 = normalize(f1), normalize(f2)
            score_mat = f1.dot(f2.T)
            if normalize_matrix:
                score_mat = normalize_matrix_rows(score_mat)
    dump_results(score_mat)

if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    result_path = caffe_root + '/data/domain_adaptation_data/results/'
    feature_1, feature_2 = 'conv3', 'conv3_p'
    feature_model = 'nordland_only_snapshots_iter_120000_10_margin.caffemodel_nordland_'
    sim_metric = 'euc || cos'
    assert sim_metric == 'euc' or sim_metric == 'cos'
    f1_npy = np.load(result_path + feature_model + feature_1 + '.npy')
    f2_npy = np.load(result_path + feature_model + feature_2 + '.npy')
    save_file = feature_model + sim_metric + '_' + feature_1 + '_' + feature_2
    main(sim_metric, f1_npy, f2_npy)