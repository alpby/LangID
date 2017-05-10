import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
import progressbar

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5):
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)

    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= np.hanning(frameSize)     # window size

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
    scale *= (freqbins-1)/max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, name, binsize=2**10, alpha=1):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)     # Use only one channel

    sshow, freq = logscale_spec(s, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    ims = np.transpose(ims)
    ims = ims[0:256,0:856] # 0-11khz, ~9s interval

    image = Image.fromarray(ims).convert('L').save(name)

# first line of traininData.csv is header (only for trainingData.csv)
file = open('../Data/Topcoder1/trainingData.csv', 'r').readlines()[1:]
os.system('mkdir ../Data/Topcoder1/trainingData/png')

for line in file:

    # Data augmention may be applied here.
    filepath = line.split(',')[0]
    os.system('mpg123 -w tmp.wav ../Data/Topcoder1/trainingData/' + filepath)
    plotstft('tmp.wav', name='../Data/Topcoder1/trainingData/png/' + filepath[:-4] + '.png', alpha=1.0)
    os.remove('tmp.wav')
