## **Language Identification using Deep Learning**

This repository is created and developed for the term project of EE578 "Speech Processing" course offered by Prof. Murat Saraçlar of Electrical and Electronics Engineering Department at Boğaziçi University in 2017 Spring Term.

Using spectrogram features is a common approach for speech processing tasks. CNNs and RNNs are used in this project to classify the speech signals according to their respective languages. The results and report will be provided on request.

### *DATASET*

Topcoder has two different datasets which are available online. You can find them at corresponding contest site.([Spoken Languages 2](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16555&compid=49304) , [Spoken Languages](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16498&pm=13845))

You can use only a part of large dataset for practical purposes. Choose number of language you want to classify before running `preprocess.py`.

### *MODULES*

To implement the methodology Python 2.7.13 is used. Below you can see the necessary modules and their versions that were used in generating and testing this code:

- Theano==0.9.0
- numpy==1.11.0
- Lasagne==0.2.dev1
- PIL==1.1.7
- scipy==0.17.1
- matplotlib==1.5.1
- Pydub==0.18.0

## *Preprocessing*

You should run `preprocess.py` first to create spectrogram of each speech sample. Once you created spectrograms you can use `--skipspectrogram` option for future operations. For large dataset input samples are created randomly. For other options please look at corresponding help sections.

## *Example Run*

After creating spectrograms you can train your models by running `main.py`. You can choose your model or which dataset to be classified. Please look at `main.py` for other options.

`python main.py --model rnn --dataset large --epochs 200 --batchsize 30 --rnnunits 100 --langs 5`
