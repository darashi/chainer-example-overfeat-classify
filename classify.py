import sys
import cv2
import numpy as np
import math
from chainer import cuda, Variable, FunctionSet, optimizers
import argparse

import overfeat_labels
import overfeat

parser = argparse.ArgumentParser(description='Chainer example')
parser.add_argument('--model', '-m', default='fast', type=str, help='model: fast, accurate')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negatives for CPU)')
parser.add_argument('files', type=str, nargs='+')
args = parser.parse_args()

if args.model == "fast":
    model = overfeat.OverFeatFast()
    model.load("net_weight_0")
elif args.model == "accurate":
    model = overfeat.OverFeatAccurate()
    model.load("net_weight_1")
else:
    parser.print_help()
    print("unsupported model: " + args.model)
    sys.exit(1)

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

def read_image(path):
  image = cv2.imread(path)
  if image == None:
      print("failed to read image: " + path)
      sys.exit(1)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  height, width, depth = image.shape

  if height < width:
      width = int(math.floor(float(width) / float(height) * model.insize))
      height = model.insize
  else:
      height = int(math.floor(float(height) / float(width) * model.insize))
      width = model.insize

  image = cv2.resize(image, (width, height))

  top = 0
  left = 0
  if height < width:
      left = int(math.floor((width - model.insize)/2))
  else:
      top = int(math.floor((height - model.insize)/2))

  image = image.transpose(2, 0, 1)

  bottom = top + model.insize
  right = left + model.insize

  image = image[:, top:bottom, left:right].astype(np.float32)
  image += -118.380948
  image /= 61.896913
  return image


for arg in args.files:
    print "---- " + arg
    image = read_image(arg)
    x_batch = np.array([image])
    if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)

    out = model.classify(x_batch)
    data = out.data

    if args.gpu >= 0:
        data = cuda.to_cpu(data)

    prediction = data[0]
    sorted_predict = sorted(range(len(prediction)),key=lambda x:prediction[x],reverse=True)

    for n in sorted_predict[0:20]:
        print overfeat_labels.labels[n], prediction[n]
