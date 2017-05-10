import sys
import numpy as np
import argparse
import json
import importlib
import os

# TODO: add argument to choose training set
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="rnn", help='Choose cnn,rnn or combine')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--mode', type=str, default="train", help='mode: train/test/test_on_train')
parser.add_argument('--batchsize', type=int, default=32, help='Number of images in one batch')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout between [0,1]')
parser.add_argument('--rnnunits', type=int, default=100, help='RNN hidden units')
parser.add_argument('--forward_cnt', type=int, default=1, help='if forward pass is nondeterministic, then how many forward passes are averaged')

args = parser.parse_args()

trndata = open("../../Data/Topcoder1/trainEqual.csv", "r").readlines()
tstdata = open("../../Data/Topcoder1/valEqual.csv", "r").readlines()

args_dict = dict(args._get_kwargs())
args_dict['trndata'] = trndata
args_dict['tstdata'] = tstdata

network_module = importlib.import_module("networks." + args.model)
model = network_module.Network(**args_dict)

print "Network specifications and training parameters are the following:"
print "Network structure is %s" % args.model
print "There are %d training examples and %d validation examples" %(len(trndata), len(tstdata))
print "Batchsize is %d" % args.batchsize
print "Dropout applied is %f" %args.dropout
print "L2 regularization applied is %f" %args.l2
if args.model == "rnn_2layers":
    print "Number of hidden RNN units is %d" % args.rnnunits

def training(mode, trndatasize, tstdatasize, epoch):
    # mode is 'train' or 'test' or 'predict'
    ygold = list()
    ypred = list()
    loss = 0
    if (mode == 'train' or mode == 'predict_on_train'):
        batches = trndatasize / args.batchsize
    elif (mode == 'test' or mode == 'predict'):
        batches = tstdatasize / args.batchsize
    else:
        raise Exception("unknown mode")

    all_prediction = list()

    for i in range(0, batches):
        print "Batch completed: %d/%d " % (i,batches)
        step_data = model.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]

        loss += current_loss
        if (mode == "predict" or mode == "predict_on_train"):
            all_prediction.append(prediction)
            for pass_id in range(args.forward_cnt-1):
                step_data = model.step(i, mode)
                prediction += step_data["prediction"]
                current_loss += step_data["current_loss"]
            prediction /= args.forward_cnt
            current_loss /= args.forward_cnt

        for x in answers:
            ygold.append(x)

        for x in prediction.argmax(axis=1):
            ypred.append(x)

    accuracy = sum([1 if t == p else 0 for t, p in zip(ygold, ypred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches / args.batchsize)

    if (mode == "predict"):
        all_prediction = np.vstack(all_prediction)
        pred_filename = "predictions/" + ("equal_split." if args.equal_split else "") + \
                         args.load_state[args.load_state.rfind('/')+1:] + ".csv"
        with open(pred_filename, 'w') as pred_csv:
            for x in all_prediction:
                print >> pred_csv, ",".join([("%.6f" % prob) for prob in x])

    return loss / batches


if args.mode == 'train':
    print "==> training"
    for epoch in range(args.epochs):
        print "Running epoch: %d/%d " % (epoch,args.epochs)
        training('train', len(trndata), len(tstdata), epoch)
        test_loss = training('test', len(trndata), len(tstdata), epoch)

elif args.mode == 'test':
    training('predict', len(trndata), len(tstdata), 0)
else:
    raise Exception("unknown mode")
