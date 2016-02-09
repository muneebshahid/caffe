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
    return score_mat#normalize_matrix(score_mat)


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
                        data[key].append([line_1[0], line_2[0]])
    return data


def dump_results(model_folder, model_id, key, result_str, results):
    np.save(save_path + model_folder + '_' + model_id + '_' + key + result_str, results)


def normalize(feature):
    flattened_features = feature.flatten().astype(dtype=np.float32)
    return flattened_features / np.linalg.norm(flattened_features)


def filter_data(key, dataset):
    filtered_data = []
    if key == 'nordland':
        for pair in dataset:
            if ('summer' in pair[0] or 'summer' in pair[1]) and ('winter' in pair[0] or 'winter' in pair[1]):
                filtered_data.append(pair)
    else:
        filtered_data = dataset
    return filtered_data

def main():
    keys = ['freiburg', 'michigan', 'nordland',]
    data = load_file(txt_path, keys)
    fe = FeatureExtractor(model_path=caffe_model_path,
                              deploy_path=deploy_path,
                              mean_binary_path=mean_binary_path,
                              input_layers=['data_1', 'data_2'])
    model_id = osh.extract_name_from_path(caffe_model_path)
    for key in data:
        print 'processing: {0}_{1}_{2}'.format(model_folder, model_id, key)
        coordinates_1, coordinates_2 = [], []
        features_1, features_2 = [], []
        key_data = filter_data(key, data[key])
        key_data_len = len(key_data)
        processed = 0
        fe.set_batch_dim(batch_size, 3, 227, 227)
        print 'total data {0}'.format(key_data_len)
        num_iter = int(np.ceil(key_data_len / float(batch_size)))
        for i in range(num_iter):
            if (batch_size * (i + 1)) <= key_data_len:
                curr_batch_size = batch_size
            else:
                curr_batch_size = key_data_len - batch_size * i
                fe.set_batch_dim(curr_batch_size)
            '''result = {'conv3': np.ones((curr_batch_size, 600)) * 5,
                      'conv3_p': np.random.rand(curr_batch_size, 600),
                      'fc8_n': np.random.rand(curr_batch_size, 128),
                      'fc8_n_p': np.random.rand(curr_batch_size, 128)}'''
            start_index = i * batch_size
            end_index = start_index + batch_size
            images = key_data[start_index:end_index]
            result = fe.extract(images=images,
                                blob_keys=['conv3', 'conv3_p'])

            features_1.extend([normalize(feature) for feature in result['conv3'].copy()])
            features_2.extend([normalize(feature) for feature in result['conv3_p'].copy()])
            coordinates_1.extend([feature for feature in result['fc8_n'].copy()])
            coordinates_2.extend([feature for feature in result['fc8_n_p'].copy()])
            processed += curr_batch_size
            print '{0} / {1}'.format(processed, len(key_data))

        print 'converting features to nd arrays...'
        features_1 = np.array(features_1)
        print features_1.shape
        features_2 = np.array(features_2)
        print 'calculating cos similarity...'
        score_mat = features_1.dot(features_2.T)
        score_mat = normalize_matrix(score_mat)
        print 'writing files...'
        dump_results(model_folder, model_id, key, '_features_1', features_1)
        dump_results(model_folder, model_id, key, '_features_2', features_2)
        dump_results(model_folder, model_id, key, '_cos_sim', score_mat)

        print 'converting coordinates to nd arrays...'
        coordinates_1 = np.array(coordinates_1)
        coordinates_2 = np.array(coordinates_2)
        print 'calculating score mat...'
        score_mat = create_score_mat(coordinates_1, coordinates_2)
        print 'writing files...'
        dump_results(model_folder, model_id, key, '_coordinates_1', coordinates_1)
        dump_results(model_folder, model_id, key, '_coordinates_2', coordinates_2)
        dump_results(model_folder, model_id, key, '_euc_dist', score_mat)
    print 'done'

if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    txt_path = caffe_root + '/data/domain_adaptation_data/images/'
    save_path = caffe_root + '/data/domain_adaptation_data/results/'
    root_model_path = caffe_root + '/data/domain_adaptation_data/models/'
    mean_binary_path = caffe_root + '../data/models/alexnet/pretrained/places205CNN_mean.binaryproto'
    model_folder = 'nordland_only'
    model_folder_path = root_model_path + model_folder + '/'
    deploy_path = model_folder_path + 'deploy.prototxt'
    caffe_model_path = model_folder_path + 'snapshots_iter_140000.caffemodel'
    batch_size = 1024
    main()
