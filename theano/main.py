# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse
import json
import importlib
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="rnn", help='Choose cnn,rnn or combine')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=32, help='Number of images in one batch')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout between [0,1]')
parser.add_argument('--rnnunits', type=int, default=100, help='RNN hidden units')
parser.add_argument('--dataset', type=str, default="small", help='Choose small or large dataset')

args = parser.parse_args()

if args.dataset == "small":
    dataPath = "../../Data/small/"
    dataclass = 5
elif args.dataset == "large":
    dataPath = "../../Data/small/"
    dataclass = 176
else:
    raise Exception('Dataset can only be small or large!')

trndata = open(dataPath + "trainEqual.csv", "r").readlines()
tstdata = open(dataPath + "valEqual.csv", "r").readlines()

args_dict = dict(args._get_kwargs())
args_dict['trndata'] = trndata
args_dict['tstdata'] = tstdata
args_dict['dataclass'] = dataclass

models = importlib.import_module("networks." + args.model)
model = models.Model(**args_dict)

print "Model specifications and training parameters are the following:"
print "Model structure is %s" % args.model
print "There are %d training examples and %d validation examples" %(len(trndata), len(tstdata))
print "Batchsize is %d" % args.batchsize
print "Dropout applied is %f" %args.dropout
print "L2 regularization applied is %f" %args.l2
if args.model == "rnn_2layers":
    print "Number of hidden RNN units is %d" % args.rnnunits

def training(mode, trndatasize, tstdatasize, epoch):

    ygold = list()
    ypred = list()
    loss = 0
    if (mode == 'train'):
        batches = trndatasize / args.batchsize
    elif (mode == 'test'):
        batches = tstdatasize / args.batchsize
    else:
        raise Exception("unknown mode")

    for i in range(0, batches):
        sys.stdout.write('\r%s %d (%s)\t |%s| %s%% %s' % ('Epoch No.', epoch+1, mode, '█' * int(50 * (i+1) // batches) + '-' * (50 - int(50 * (i+1) // batches)), ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(batches))), 'Complete'))

        step_data = model.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]

        loss += current_loss
        for x in answers:
            ygold.append(x)

        for x in prediction.argmax(axis=1):
            ypred.append(x)

        sys.stdout.flush()

    accuracy = sum([1 if t == p else 0 for t, p in zip(ygold, ypred)])
    print ":----->\t%s accuracy: %f%%" % (mode, accuracy * 100.0 / batches / args.batchsize)

    return loss / batches

for epoch in range(args.epochs):
    training('train', len(trndata), len(tstdata), epoch)
    test_loss = training('test', len(trndata), len(tstdata), epoch)
