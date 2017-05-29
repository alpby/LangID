# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib as mpl
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from PIL.Image import fromarray
import os
import argparse
from pydub import AudioSegment
import warnings
mpl.use('TkAgg')

def STFT(data, frameSize):
    # NOTE: Overlapping factor is can be changed below. 0.5 is default value.
    hopSize = int(frameSize - np.floor(0.5 * frameSize))

    samples = np.append(np.zeros(int(frameSize/2.0)), data)
    freqs = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples,shape=(freqs, frameSize),strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()

    frames *= np.hanning(frameSize)
    return np.fft.rfft(frames)

def logSpectrum(spec):
    _, freqbins = spec.shape
    scale = range(freqbins)
    logspec = np.complex128(np.zeros(spec.shape))

    for i in range(freqbins - 1):
        logspec[:,i] = np.sum(spec[:,scale[i]:scale[i + 1]], axis=1)
    logspec[:,freqbins - 1] = np.sum(spec[:,scale[freqbins - 1]:], axis=1)

    return logspec

# NOTE: The number of frequency bins are different from one to another sample. So they had to be cropped and become the same to train the model. The shape of (256,800) is reasonable in that case.
# TODO: Shape of spectrograms can be changed to decrease training time. Especially if the properties of languages to be classified is well-known, for instance they can be distinguished by looking at certain frequencies, only the frequencies that matter can be cropped and spectrograms can be created accordingly.
def spectrograms(audiopath, name):
    _, samples = wav.read(audiopath)
    s = STFT(samples, 1024)     # Use only one channel

    logspec = logSpectrum(s)
    spectrogram = dB(logspec,10e-6)

    spectrogram = np.transpose(spectrogram)
    spectrogram = fromarray(spectrogram[0:256,0:800]).convert('L').save(name)

def dB(data, ref):
    return 20*np.log10(np.abs(data) / ref)

# TODO: Data augmentation may be applied to get more data.
# There is no mp3 reader in Python.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")     # Escape log(0) warning.

    parser = argparse.ArgumentParser()

    # NOTE: I chose the ratio between number of samples in training and test set to be 0.8 in this project.
    parser.add_argument('--ratio', type=float, default=0.8, help='Training and test set ratio')
    parser.add_argument('--dataset', type=str, default="small", help='Choose small or large dataset')

    args = parser.parse_args()

    # NOTE: There are two different datasets.
    # The small one has 120 samples from 5 different European languages(English, German, Spanish, Italian and French).
    # The large one has 376 samples from 176 different languages.
    if args.dataset == "small":
        datapath = "../Data/small/"
    elif args.dataset == "large":
        datapath = "../Data/large/"
    else:
        raise Exception('Dataset can only be small or large!')

    if not os.path.exists(datapath + "data/spectrograms"):
        os.system("mkdir " + datapath + "data/spectrograms")

    csvfile = open(datapath + "data.csv", 'r').readlines()
    langs = dict()
    langCount = 0

    i = 0
    for line in csvfile:
        # NOTE: Fancy progress bar
        sys.stdout.write('\r%s |%s| %s%% %s' % ('Saving spectrograms...', '█' * int(50 * (i+1) // len(csvfile)) + '-' * (50 - int(50 * (i+1) // len(csvfile))), ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(len(csvfile)))), 'Complete'))

        filepath,lang = line.split(",")
        AudioSegment.from_mp3(datapath + 'data/' + filepath).export(filepath + ".wav", format="wav")
        spectrograms(filepath + ".wav", name = datapath + "data/spectrograms/" + filepath[:-4] + ".png")
        os.remove(filepath + ".wav")
        lang = lang.strip()
        if lang not in langs:
            langs[lang] = langCount
            langCount += 1

        i += 1
        sys.stdout.flush()
    print "\n"

    # NOTE: Dataset is splitted into training and test sets. Sound files' names and labels are also saved in different .csv files.
    trainSet = open(datapath + "trainset.csv", 'w')
    testSet = open(datapath + "testset.csv", 'w')
    count = [0] * langCount
    for line in csvfile:
        filepath, language = line.split(",")
        lang = langs[language.strip()]
        if (count[lang] < (len(csvfile)/len(langs)) * args.ratio):
            trainSet.write(filepath[:-4] + ',' + str(lang) + '\n')
        else:
            testSet.write(filepath[:-4] + ',' + str(lang) + '\n')
        count[lang] += 1
