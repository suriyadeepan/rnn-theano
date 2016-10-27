import itertools
import csv
import nltk

import numpy as np
import pickle as p

vocab_size = 8000
token_unknown = 'UNK'
token_start = 'START'
token_end = 'END'

'''
* Read the data from csv file
* append SENTENCE_START and SENTENCE_END tokens
'''
def read_data(file_name='data/reddit-comments-2015-08.csv'):
    with open(file_name) as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.__next__()
        # split comments into sentences
        sentences = itertools.chain(*[ nltk.sent_tokenize(x[0]) for x in reader ])
        # add tokens - [start,end]
        sentences = [ '{0} {1} {2}'.format(token_start,x,token_end) for x in sentences ]
        # tokenize sentences
        return [ nltk.word_tokenize(sent) for sent in sentences ]

'''
    Need
    1. vocab
    2. index2word
    3. word2index
'''
def gen_vocab(tokenized_sentences):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 8000 most used words
    vocab = freq_dist.most_common(vocab_size-1)
    # index2word
    index2word = [ x[0] for x in vocab ]
    # append unknown
    index2word.append(token_unknown)
    # word2index
    word2index = dict( [(w,i) for i,w in enumerate(index2word)] )
    return vocab, index2word, word2index

'''
    Add UNK tokens to sentences
'''
def tokenize_unk(tokenized_sentences, index2word):
    for i,sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [ w if w in index2word else token_unknown for w in sent ]
    return tokenized_sentences

'''
    X_train, Y_train
'''
def gen_dataset(tokenized_sentences, word2index):
    X_train = np.asarray( [ [word2index[w] for w in sent[:-1] ] for sent in tokenized_sentences])
    Y_train = np.asarray( [ [word2index[w] for w in sent[1:] ] for sent in tokenized_sentences])
    return X_train, Y_train

'''
    MAIN function
'''
def execute(data_file = 'data/reddit-comments-2015-08.csv'):
    # read file -> tokenized sentences
    sentences = read_data(data_file)
    # create vocabulary
    vocab, index2word, word2index = gen_vocab(sentences)
    # add unk to sentences
    tokenize_unk(sentences, index2word)
    # generate datset
    X_train, Y_train = gen_dataset(sentences, word2index)
    data = {'vocab' : vocab, 'word2index' : word2index, 'index2word' : index2word, 'X_train' : X_train, 'Y_train' : Y_train}
    # save to disk
    #   utils.pickle_data(data)
    return data
