import caffe
import numpy as np
import os_helper as osh


def forward(net, transformer, img1, img2):
    img1 = transformer.preprocess(transformer_key, caffe.io.load_image(img1))
    img2 = transformer.preprocess(transformer_key, caffe.io.load_image(img2))
    print img1.shape
    net.blobs['data_1'].data[...] = img1
    net.blobs['data_2'].data[...] = img2
    output = net.forward()
    return output['fc8_n'], output['fc8_n_p']


def print_r(net, img1, img2):
    r1 = output(net, img1, img2)
    print '-----------------------------------'
    print "untrained: ", r1
    print 'distance: ', np.linalg.norm(r1['fc8_n'] - r1['fc8_n_p'])


def create_transformer(net, mean_arr):
    transformer = caffe.io.Transformer({transformer_key: net.blobs['data_1'].data.shape})
    transformer.set_transpose(transformer_key, (2, 0, 1))
    transformer.set_mean(transformer_key, mean_arr[0].mean(1).mean(1))
    transformer.set_raw_scale(transformer_key, 255)
    transformer.set_channel_swap(transformer_key, (2, 1, 0))
    return transformer


def load_mean_binary():
    # load mean binary proto
    blob = caffe.proto.caffe_pb2.BlobProto()
    mean_data = open(data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto', 'rb').read()
    blob.ParseFromString(mean_data)
    return np.array(caffe.io.blobproto_to_array(blob))


def load_test_image_txt():
    test_files = ['test1.txt', 'test2.txt']
    ims = [[], []]
    for i, test_file in enumerate(test_files):
        with open(image_txt + test_file, 'r') as file_handle:
            for line in file_handle.readlines():
                col = line.replace('\n', '').split(' ')
                if col[1] == '1':
                    ims[i].append(col[0])
    return [[im1, im2] for im1, im2 in zip(ims[0], ims[1])]

def main():
    net = caffe.Net(deploy_prototxt, caffe_model, caffe.TEST)
    transformer = create_transformer(net, load_mean_binary())
    arr = load_test_image_txt()
    coordinates = [forward(net, transformer, pair[0], pair[1]) for pair in arr]
    return


if __name__ == '__main__':
    caffe.set_mode_gpu()
    transformer_key = 'data_'
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    data = caffe_root + '/../data/'
    results = caffe_root + '/../results/'
    image_txt = caffe_root + '/data/domain_adaptation_data/images/'
    deploy_prototxt = caffe_root + '/examples/domain_adaptation/network/alexnet/train_vals/deploy.prototxt'
    caffe_model = results + '/alex/1-200k/snapshots_iter_200000.caffemodel'
    mean_file = data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto'
    main()