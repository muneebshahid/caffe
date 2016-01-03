import caffe
import numpy as np
import os_helper as osh

caffe.set_mode_gpu()
caffe_root = osh.get_env_var('CAFFE_ROOT')
data = caffe_root + '../data/'
mean_file = data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto'
im1 = data + 'images/freiburg/uncompressed/summer/imageCompressedCam0_0000002.jpg'
im2 = data + 'images/freiburg/uncompressed/winter/imageCompressedCam0_0001645.jpg'
blob = caffe.proto.caffe_pb2.BlobProto()
data = open('../data/models/alexnet/pretrained/places205CNN_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
net1 = caffe.Net(caffe_root + '/examples/domain_adaptation/\
								network/alexnet/train_vals/siamese/deploy.prototxt', caffe.TEST)
net2 = caffe.Net(caffe_root + '/examples/domain_adaptation/\
								network/alexnet/train_vals/siamese/deploy.prototxt',
								'examples/domain_adaptation/snapshots/snapshots_iter_50000.caffemodel',
								caffe.TEST)

def output(net, im1, im2):
    transformer = caffe.io.Transformer({'source_target': net.blobs['source_target_1'].data.shape})
    transformer.set_transpose('source_target', (2, 0, 1))
    transformer.set_mean('source_target', arr[0].mean(1).mean(1))
    transformer = caffe.io.Transformer({'source_target': net.blobs['source_target_1'].data.shape})
    transformer.set_transpose('source_target', (2, 0, 1))
    transformer.set_mean('source_target', arr[0].mean(1).mean(1))
    transformer.set_raw_scale('source_target', 255)
    transformer.set_channel_swap('source_target', (2, 1, 0))
    net.blobs['source_target_1'].data[...] = transformer.preprocess('source_target_1', caffe.io.load_image(im1))
    net.blobs['source_target_2'].data[...] = transformer.preprocess('source_target_2', caffe.io.load_image(im2))
    return net.forward()

def print_r():
	r1 = output(net1, im1, im2)
	r2 = output(net2, im1, im2)
	print '-----------------------------------'
	print r1, r2
	print '-----------------------------------'