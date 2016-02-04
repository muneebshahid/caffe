from extract_features import FeatureExtractor
import numpy as np


def load_file(path):
    data = []
    with open(path) as file_handle:
        data = [line.replace('\n', '') for line in file_handle.readlines()]
    return data


def normalize(feature):
    return feature.flatten().astype(dtype=np.float64)[np.newaxis, :] / np.linalg.norm(feature)


def main(txt_path, save_path, model_path, deploy_path, mean_binary_path):
    fe = FeatureExtractor(model_path, deploy_path, mean_binary_path)
    data_1 = load_file(txt_path[0])
    data_2 = load_file(txt_path[1])
    features = [np.empty((len(data_1), 64896)), np.empty((len(data_2), 64896))]
    for i, im1, im2 in enumerate(zip(data_1, data_2)):
        result = fe.extract(im1, im2)
        temp_features = [result['conv3'].copy(), result['conv3_p'].copy()]
        for j, temp_feature in enumerate(temp_features):
            features[i][j] = normalize(temp_feature)
    for f, feature in enumerate(features):
        np.savetxt(save_path + str(f) + '.npy', feature)

if __name__ == '__main__':
    main()
