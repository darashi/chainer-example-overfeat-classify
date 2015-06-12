from chainer import FunctionSet, Variable
import chainer.functions as F
import numpy as np

class OverFeatParameterLoader:
    def __init__(self, path):
        self.weights = np.fromfile(path, dtype=np.float32)
        self.pos = 0

    def load_layer(self, conv):
        w, b = conv.parameters
        nw = reduce(lambda x, y: x * y, w.shape)
        nb = b.shape[0]
        np.copyto(w, self.weights[self.pos:self.pos+nw].reshape(w.shape))
        self.pos += nw
        np.copyto(b, self.weights[self.pos:self.pos+nb])
        self.pos += nb

class OverFeatFast(FunctionSet):

    insize = 231

    def __init__(self):
        super(OverFeatFast, self).__init__(
                conv1 = F.Convolution2D(   3,   96, 11, stride=4),
                conv2 = F.Convolution2D(  96,  256,  5),
                conv3 = F.Convolution2D( 256,  512,  3, pad=1),
                conv4 = F.Convolution2D( 512, 1024,  3, pad=1),
                conv5 = F.Convolution2D(1024, 1024,  3, pad=1),
                conv6 = F.Convolution2D(1024, 3072,  6),
                fc7   = F.Linear(3072, 4096),
                fc8   = F.Linear(4096, 1000),
                )

    def load(self, path):
        loader = OverFeatParameterLoader(path)
        loader.load_layer(self.conv1)
        loader.load_layer(self.conv2)
        loader.load_layer(self.conv3)
        loader.load_layer(self.conv4)
        loader.load_layer(self.conv5)
        loader.load_layer(self.conv6)
        loader.load_layer(self.fc7)
        loader.load_layer(self.fc8)

    def classify(self, x_data):
        x = Variable(x_data, volatile=True)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
        h = F.relu(self.conv6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return F.softmax(h)

class OverFeatAccurate(FunctionSet):

    insize = 221

    def __init__(self):
        super(OverFeatAccurate, self).__init__(
                conv1 = F.Convolution2D(   3,   96, 7, stride=2),
                conv2 = F.Convolution2D(  96,  256, 7),
                conv3 = F.Convolution2D( 256,  512, 3, pad=1),
                conv4 = F.Convolution2D( 512,  512, 3, pad=1),
                conv5 = F.Convolution2D( 512, 1024, 3, pad=1),
                conv6 = F.Convolution2D(1024, 1024, 3),
                conv7 = F.Convolution2D(1024, 4096, 5),
                fc8   = F.Linear(4096, 4096),
                fc9   = F.Linear(4096, 1000),
                )

    def load(self, path):
        loader = OverFeatParameterLoader(path)
        loader.load_layer(self.conv1)
        loader.load_layer(self.conv2)
        loader.load_layer(self.conv3)
        loader.load_layer(self.conv4)
        loader.load_layer(self.conv5)
        loader.load_layer(self.conv6)
        loader.load_layer(self.conv7)
        loader.load_layer(self.fc8)
        loader.load_layer(self.fc9)

    def classify(self, x_data):
        x = Variable(x_data, volatile=True)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=3)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 3, stride=3)
        h = F.relu(self.conv7(h))
        h = F.relu(self.fc8(h))
        h = self.fc9(h)
        return F.softmax(h)
