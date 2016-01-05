import caffe
import numpy as np
import os_helper as osh

caffe.set_mode_gpu()
caffe_root = osh.get_env_var('CAFFE_ROOT')
data = caffe_root + '/../data/'
mean_file = data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto'
im1 = data + 'images/freiburg/uncompressed/summer/imageCompressedCam0_0000002.jpg'
im2 = data + 'images/freiburg/uncompressed/winter/imageCompressedCam0_0001645.jpg'
blob = caffe.proto.caffe_pb2.BlobProto()
mean_data = open(data + 'models/alexnet/pretrained/places205CNN_mean.binaryproto', 'rb').read()
blob.ParseFromString(mean_data)
arr = np.array(caffe.io.blobproto_to_array(blob))
net1 = caffe.Net(caffe_root + '/examples/domain_adaptation/network/alexnet/pretrained/train_vals/siamese/deploy.prototxt', caffe.TEST)
net2 = caffe.Net(caffe_root + '/examples/domain_adaptation/network/alexnet/pretrained/train_vals/siamese/deploy.prototxt',
		caffe_root + '/examples/domain_adaptation/snapshots/snapshots_iter_50000.caffemodel',
			caffe.TEST)

def output(net, img1, img2):
	transformer = caffe.io.Transformer({'source_target': net.blobs['source_target_1'].data.shape})
	transformer.set_transpose('source_target', (2, 0, 1))
	transformer.set_mean('source_target', arr[0].mean(1).mean(1))
	transformer.set_raw_scale('source_target', 255)
	transformer.set_channel_swap('source_target', (2, 1, 0))
	img1 = transformer.preprocess('source_target', caffe.io.load_image(img1))
	img2 = transformer.preprocess('source_target', caffe.io.load_image(img2))
	#img1 = caffe.io.oversample(img1, (227, 227))
	#iimg2 = caffe.io.oversample(img2_, (227, 227))	
	net.blobs['source_target_1'].data[...] = img1
     	net.blobs['source_target_2'].data[...] = img2
    	return net.forward()


def print_r(img1, img2):
	r1 = output(net1, img1, img2)
	r2 = output(net2, img1, img2)
	print '-----------------------------------'
	print "untrained: ", r1 
	print 'distance: ', np.linalg.norm(r1['fc8_n'] - r1['fc8_n_p'])
	print '-----------------------------------'
	print "untrained: ", r2 
	print 'distance: ', np.linalg.norm(r2['fc8_n'] - r2['fc8_n_p'])

