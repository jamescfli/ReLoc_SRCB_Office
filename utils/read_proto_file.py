import caffe
import numpy as np


blob = caffe.proto.caffe_pb2.BlobProto()
data = open('proto_files/places365CNN_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
print arr