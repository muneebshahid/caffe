import numpy as np
import caffe


class FeatureExtractor():
    __net = None
    __transformer = None
    __transformer_key = 'data_'

    def __init__(self, model_path, deploy_path, mean_binary_path):

        # convert binary proto to numpy array
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(open(mean_binary_path, 'rb').read())
        mean_data = np.array(caffe.io.blobproto_to_array(blob))

        # create net instance.
        self.__net = caffe.Net(deploy_path, model_path, caffe.TEST)
        # create data transformer.
        self.__transformer = caffe.io.Transformer({self.__transformer_key: self.__net.blobs['data_1'].data.shape})
        # swap channels with width and height
        self.__transformer.set_transpose(self.__transformer_key, (2, 0, 1))
        self.__transformer.set_mean(self.__transformer_key, mean_data[0].mean(1).mean(1))
        # set max scale i.e  0 - 255
        self.__transformer.set_raw_scale(self.__transformer_key, 255)
        # switch to bgr from rgb
        self.__transformer.set_channel_swap(self.__transformer_key, (2, 1, 0))
        return

    def extract(self, images, blob_keys, input_layers=('data_1', 'data_2')):
        assert len(images) == len(input_layers)
        for i, input_layer in enumerate(input_layers):
            image = self.__transformer.preprocess(self.__transformer_key, caffe.io.load_image(images[i]))
            self.__net.blobs[input_layer].data[...] = image
        return self.__net.forward(blob_keys)
