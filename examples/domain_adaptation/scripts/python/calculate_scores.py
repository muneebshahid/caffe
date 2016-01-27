import caffe
import numpy as np
import os_helper as osh

caffe.set_mode_gpu()
caffe_root = osh.get_env_var('CAFFE_ROOT')
data = caffe_root + '/../data/'
results = caffe_root + '/../results/'
mean_file = data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto'
<<<<<<< HEAD
im1 = data + 'images/orig/freiburg/summer/imageCompressedCam0_0000002.jpg'
im2 = data + 'images/orig/freiburg/winter/imageCompressedCam0_0001645.jpg'
deploy = '/examples/domain_adaptation/network/alexnet/pretrained/train_vals/deploy.prototxt'
=======
im1 = data + 'images/orig/freiburg/summer/imageCompressedCam0_0003904.jpg'
im2 = data + 'images/orig/freiburg/winter/imageCompressedCam0_0016412.jpg'
>>>>>>> aac5fcbe052aa62e4fcc3c208013430948530e87
blob = caffe.proto.caffe_pb2.BlobProto()
mean_data = open(data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto', 'rb').read()
caffe_model = results + '/alex/1-200k/snapshots_iter_200000.caffemodel'
blob.ParseFromString(mean_data)
arr = np.array(caffe.io.blobproto_to_array(blob))
<<<<<<< HEAD
net1 = caffe.Net(caffe_root + deploy,
                 caffe.TEST)
net2 = caffe.Net(caffe_root + deploy,
                 results + '/alex/1-200k/snapshots_iter_200000.caffemodel', caffe.TEST)
=======
net1 = caffe.Net(caffe_root +
                 '/examples/domain_adaptation/network/alexnet/pretrained/train_vals/deploy.prototxt',
                 caffe.TEST)
net2 = caffe.Net(caffe_root +
                 '/examples/domain_adaptation/network/alexnet/pretrained/train_vals/deploy.prototxt',
                 caffe_model, caffe.TEST)
>>>>>>> aac5fcbe052aa62e4fcc3c208013430948530e87


def output(net, img1, img2):
    transformer = caffe.io.Transformer({'data_': net.blobs['data_1'].data.shape})
    transformer.set_transpose('data_', (2, 0, 1))
    transformer.set_mean('data_', arr[0].mean(1).mean(1))
    transformer.set_raw_scale('data_', 255)
    transformer.set_channel_swap('data_', (2, 1, 0))
    img1 = transformer.preprocess('data_', caffe.io.load_image(img1))
    img2 = transformer.preprocess('data_', caffe.io.load_image(img2))
    net.blobs['data_1'].data[...] = img1
    net.blobs['data_2'].data[...] = img2
    return net.forward()


def print_r(img1, img2):
    r1 = output(net1, img1, img2)
    r2 = output(net2, img1, img2)
    print '-----------------------------------'
    print "untrained: ", r1
    print 'distance: ', np.linalg.norm(r1['fc8_n'] - r1['fc8_n_p'])
    print '-----------------------------------'
    print "trained: ", r2
    print 'distance: ', np.linalg.norm(r2['fc8_n'] - r2['fc8_n_p'])

