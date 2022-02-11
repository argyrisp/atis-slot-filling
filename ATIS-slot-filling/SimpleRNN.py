import pickle
import numpy as np
import time
import random
import os


from keras.models import Sequential
from keras.layers import (Input, Embedding, SimpleRNN, Dense, Activation,
                          TimeDistributed, Convolution1D, Dropout, GRU)
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from metrics.accuracy import conlleval
from utils.tools import shuffle

folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder): os.mkdir(folder)

s = {'fold':0, # 5 folds 0,1,2,3,4
         'verbose':0, # flag for real time printing, suggested 0
         'nhidden':100, # number of hidden units
         'seed':345, # used for shuffling while training
         'emb_dimension':100, # dimension of word embedding
         'nepochs':30} # total epochs for training


fname = 'atis.fold' + str(s['fold']) + '.pkl'

# unpickle data set
with open(fname, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set, dicts = u.load()

# create index to word dictionaries
idx2label = dict((k,v) for v,k in dicts['labels2idx'].items())
idx2word = dict((k,v) for v,k in dicts['words2idx'].items())

# split data to samples, labels, x_ne is labels flipped ie x_y: [126 126 0 126] is x_ne:  [0 0 126 0], not used
train_lex, train_ne, train_y = train_set
valid_lex, valid_ne, valid_y = valid_set
test_lex,  test_ne,  test_y  = test_set

# define some useful parameters
vocsize = len(dicts['words2idx'])   # how many different words there are
nclasses = len(dicts['labels2idx']) # how many labels there are
nsentences = len(train_lex)         # how many samples there are

# randomize
np.random.seed(s['seed'])
random.seed(s['seed'])

# model initialization
model = Sequential()
model.add(Embedding(vocsize, s['emb_dimension']))                   # default 100 dimension
model.add(Dropout(0.25))                                            # drop out with 25% chance
model.add(SimpleRNN(100, return_sequences=True))                    # RNN layer with 100 hidden units
model.add(TimeDistributed(Dense(nclasses, activation='softmax')))   # softmax activation function
model.compile('rmsprop', 'categorical_crossentropy')                # compiler

# loop for training
best_f1 = -np.inf
print('Training and evaluation begins, please wait...')
tic1 = time.time()
total_training_time = 0
for e in range(s['nepochs']):
    shuffle([train_lex, train_ne, train_y], s['seed'])
    s['ce'] = e+1
    tic = time.time()
    for i in range(nsentences):
        X = np.asarray([train_lex[i]])
        Y = to_categorical(np.asarray(train_y[i])[:, np.newaxis], nclasses)[np.newaxis, :, :]

        if X.shape[1] == 1:     # skip iteration due to possible bug
            continue
        model.train_on_batch(X, Y)      # train on current sentence with X as the sentence and Y as the label

        if s['verbose'] == 1:
            print('[learning] epoch %i >> %2.2f%%' % (s['ce'], (i + 1) * 100. / nsentences), 'completed in %.2f (sec)' % (time.time()-tic)) #, 'completed in %.2f (sec) <<\r' % (time.time() - tic), sys.stdout.flush())
    print('epoch %i >> completed in %.2f (sec)' % (s['ce'], (time.time()-tic)))
    total_training_time += time.time()-tic

    predictions_test = [map(lambda x: idx2label[x], \
                            model.predict_on_batch(\
                                np.asarray([x])).argmax(2)[0])\
                        for x in test_lex]                                          # predict on test
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]
    predictions_valid = [map(lambda x: idx2label[x], \
                             model.predict_on_batch( \
                                 np.asarray([x])).argmax(2)[0]) \
                         for x in valid_lex]
    groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

    res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt') # compute f1
    #res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

    if res_test['f1'] > best_f1:                                    # check if current epoch is best and save weights
        model.save_weights('best_model.h5', overwrite=True)
        best_f1 = res_test['f1']

        print('NEW BEST: epoch', s['ce'], 'best test F1', res_test['f1'], ' ' * 20)
        #s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
        s['tf1'], s['tp'], s['tr'] = res_test['f1'], res_test['p'], res_test['r']
        s['be'] = s['ce']

print('BEST RESULT: epoch', s['be'], 'best test F1', s['tf1'], 'with the model', folder)
print('Process completed in %.2f (sec), %.2f (sec/epoch)' % (time.time()-tic1, total_training_time/s['nepochs']))
