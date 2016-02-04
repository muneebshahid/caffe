import numpy as np
import caffe


class FeatureExtractor():
    __net = None
    __transformer = None

    def __init__(self, model_path, deploy_path, mean_binary_path):

        # convert binary proto to numpy array
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(open(mean_binary_path, 'rb').read())
        mean_data = np.array(caffe.io.blobproto_to_array(blob))

        # create net instance.
        self.__net = caffe.Net(deploy_path, model_path, caffe.TEST)
        # create data transformer.
        self.__transformer = caffe.io.Transformer({'data_': self.__net.blobs['data_1'].data.shape})
        # swap channels with width and height
        self.__transformer.set_transpose('data_', (2, 0, 1))
        self.__transformer.set_mean('data_', mean_data[0].mean(1).mean(1))
        # set max scale i.e  0 - 255
        self.__transformer.se_raw_scale('data_', 255)
        # switch to bgr from rgb
        self.__transformer.set_channel_swap('data_', (2, 1, 0))
        return

    def extract(self, image_pair, blob_keys):
        img1 = self.__transformer.preprocess('data_', caffe.io.load_image(image_pair[0]))
        img2 = self.__transformer.preprocess('data_', caffe.io.load_image(image_pair[1]))
        self.__net.blobs['data_1'].data[...] = img1
        self.__net.blobs['data_2'].data[...] = img2
        return self.__net.forward(blob_keys)