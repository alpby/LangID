import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
import argparse

def stft(sig, frameSize, overlapFac=0.5):
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    samples = np.append(np.zeros(int(frameSize/2.0)), sig)
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples,shape=(cols, frameSize),strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()

    frames *= np.hanning(frameSize)

    return np.fft.rfft(frames)

def logSpectrum(spec, sr=44100):
    # spec = spec[:, 0:120]
    timebins, freqbins = spec.shape

    scale = np.linspace(0, 1, freqbins)
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = list()
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

def plotSpectrogram(audiopath, name, binsize=2**10):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)     # Use only one channel

    sshow, freq = logSpectrum(s, sr=samplerate)
    sshow = sshow[2:, :]
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    ims = np.transpose(ims)
    image = Image.fromarray(ims[0:256,0:800]).convert('L').save(name)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="small", help='Choose small or large dataset')
parser.add_argument('--ratio', type=float, default=0.8, help='Training and test set ratio')

args = parser.parse_args()
if args.dataset == "small":
    datapath = "../Data/small/"
elif args.dataset == "large":
    datapath = "../Data/large/"
else:
    raise Exception('Dataset can only be small or large!')

os.system("mkdir " + datapath + "trainingData/spectrograms")

csvfile = open(datapath + "trainingData.csv", 'r').readlines()
trainSet = open(datapath + "trainset.csv", 'w')
testSet = open(datapath + "testset.csv", 'w')

langs = dict()
langCount = 0

for line in csvfile:
    # TODO: Data augmentation may be applied to get more data.
    filepath,lang = line.split(",")
    os.system('mpg123 -w tmp.wav ' + datapath + 'trainingData/' + filepath)
    plotSpectrogram('tmp.wav', name = datapath + "trainingData/spectrograms/" + filepath[:-4] + ".png")
    os.remove('tmp.wav')
    lang = lang.strip()
    if lang not in langs:
        langs[lang] = langCount
        langCount += 1

count = [0] * langCount
for line in csvfile:
    filepath, language = line.split(",")
    lang = langs[language.strip()]
    if (count[lang] < (len(csvfile)/len(langs))*args.ratio):
        trainSet.write(filepath[:-4] + ',' + str(lang) + '\n')
    else:
        testSet.write(filepath[:-4] + ',' + str(lang) + '\n')
    count[lang] += 1
