from extract_features import FeatureExtractor
import numpy as np
import os_helper as osh


def normalize_matrix(matrix):
    return np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, matrix)


def create_score_mat(qu, db):
    score_mat = np.ones((qu.shape[0], qu.shape[0]))
    for i, qu_point in enumerate(qu):
        for j, db_point in enumerate(db):
            score_mat[i, j] = np.linalg.norm(qu_point - db_point)
        if i % 100 == 0:
            print i
    return normalize_matrix(score_mat)


def load_file(path, keys):
    data = {}
    for key in keys:
        data[key] = []
        with open(path + '/test1.txt') as test_1, open(path + '/test2.txt') as test_2:
            for line_1, line_2 in zip(test_1.readlines(), test_2.readlines()):
                line_1 = line_1.replace('\n', '').split(' ')
                line_2 = line_2.replace('\n', '').split(' ')
                if line_1[1] == '1' and line_2[1] == '1':
                    if key in line_1[0] and key in line_2[0]:
                        data[key].append([line_1, line_2])
    return data


def normalize(feature):
    flattened_features = feature.flatten().astype(dtype=np.float32)
    return flattened_features / np.linalg.norm(flattened_features)


def main():
    keys = ['freiburg', 'michigan']
    fe = FeatureExtractor(model_path, deploy_path, mean_binary_path)
    data = load_file(txt_path, keys)
    for key in data:
        print 'processing: ', key
        coordinates_1, coordinates_2 = [], []
        features_1, features_2 = [], []
        key_data = data[key]
        for i, image_pair in enumerate(key_data):
            im1, im2 = image_pair[0], image_pair[1]
            result = fe.extract([im1, im2], ['conv3', 'conv3_p'])
            features_1.append(normalize(result['conv3'].copy()))
            features_2.append(normalize(result['conv3_p'].copy()))
            coordinates_1.append(result['fc8_n'][0].copy())
            coordinates_2.append(result['fc8_n_p'][0].copy())
            if i % 500 == 0:
                print '{0} / {1}'.format(i, len(key_data))

        print 'converting features to nd arrays...'
        features_1 = np.array(features_1)
        features_2 = np.array(features_2)
        print 'calculating cos similarity...'
        score_mat = features_1.dot(features_2.T)
        score_mat = normalize_matrix(score_mat)
        print 'writing files...'
        np.save(save_path + '_' + key + '_features_1', np.array(features_1))
        np.save(save_path + '_' + key + '_features_2', np.array(features_2))
        np.save(save_path + '_' + key + '_cos_sim', np.array(score_mat))

        print 'converting coordinates to nd arrays...'
        coordinates_1 = np.array(coordinates_1)
        coordinates_2 = np.array(coordinates_2)
        print 'calculating score mat...'
        score_mat = create_score_mat(coordinates_1, coordinates_2)
        print 'writing files...'
        np.save(save_path + '_' + key + '_coordinates_1', np.array(coordinates_1))
        np.save(save_path + '_' + key + '_coordinates_2', np.array(coordinates_2))
        np.save(save_path + '_' + key + '_euc_dist', np.array(score_mat))
    print 'done'

if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    txt_path = caffe_root + '/data/domain_adaptation_data/images/'
    save_path = caffe_root + '/data/domain_adaptation_data/results/'
    deploy_path = caffe_root + '/examples/domain_adaptation/network/deploy.prototxt'
    model_path = caffe_root + '../results/curr.caffemodel'
    mean_binary_path = caffe_root + '../data/models/alexnet/pretrained/places205CNN_mean.binaryproto'
    main()
