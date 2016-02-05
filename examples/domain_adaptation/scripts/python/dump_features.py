from extract_features import FeatureExtractor
import numpy as np
import os_helper as osh


def load_file(path):
    data = []
    with open(path) as file_handle:
        for line in file_handle.readlines():
            line = line.replace('\n', '').split(' ')
            if line[1] == '1':
                data.append(line[0])
    return data


def normalize(feature):
    flattened_features = feature.flatten().astype(dtype=np.float32)
    return flattened_features / np.linalg.norm(flattened_features)


def main():
    fe = FeatureExtractor(model_path, deploy_path, mean_binary_path)
    data_1 = load_file(txt_path + '/test1.txt')
    data_2 = load_file(txt_path + '/test2.txt')
    features_1, features_2 = [], []
    for i, (im1, im2) in enumerate(zip(data_1, data_2)):
        #result = {'conv3': np.random.rand(64896), 'conv3_p': np.random.rand(64896)}#fe.extract([im1, im2], ['conv3', 'conv3_p'])
        result = fe.extract([im1, im2], ['conv3', 'conv3_p'])
        features_1.append(normalize(result['conv3'].copy()))
        features_2.append(normalize(result['conv3_p'].copy()))
        if i % 100 == 0:
            print '{0} / {1}'.format(i, len(data_1))
    print 'writing files....'
    np.save(save_path + 'features_1', np.array(features_1))
    np.save(save_path + 'features_2', np.array(features_2))


if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    txt_path = caffe_root + '/data/domain_adaptation_data/images/'
    save_path = caffe_root + '/data/domain_adaptation_data/features/'
    deploy_path = caffe_root + '/examples/domain_adaptation/network/deploy.prototxt'
    model_path = caffe_root + '../results/curr.caffemodel'
    mean_binary_path = caffe_root + '../data/models/alexnet/pretrained/places205CNN_mean.binaryproto'
    main()
