import pickle as p
import numpy as np
from gen_data import token_unknown, token_start, token_end



'''
    pickle dataset
'''
def pickle_data(data, filename='data/pdata.pkl'):
    with open(filename, 'wb') as f:
        p.dump(data,f)

'''
    read from pickle
'''
def read_pickle(pkl_file = 'data/pdata.pkl'):
    with open(pkl_file,'rb') as f:
        return p.load(f)


'''
    indexed sentence to "wordy" sentence?
'''
def decode_sentence(npsent, index2word, word2index):
    tokens = [ word2index[tok] for tok in [token_end, token_start, token_unknown] ]
    return ' '.join([ index2word[i] for i in npsent if i not in tokens ])
    #return ' '.join([ index2word[i] for i in npsent])

'''
    list of words to list indices
'''
def encode_list(wlist, word2index):
    return [ word2index[w] for w in wlist ]

'''
    save model to file
'''
def save(model, filename):
    model_params = { 'U' : model.U.get_value(), 'V' : model.V.get_value(), 'W' : model.W.get_value() }
    pickle_data(model_params, filename=filename)
    print('\nModel saved to {}'.format(filename))

'''
    load model from file
'''
def load(model, filename):
    model_params = read_pickle(filename)
    assert model_params['U'].shape[0] == model.hidden_dim
    assert model_params['U'].shape[1] == model.word_dim
    model.U.set_value(model_params['U'])
    model.V.set_value(model_params['V'])
    model.W.set_value(model_params['W'])
    print('\nModel loaded from {}'.format(filename))

'''
    load model from npz file
'''
def load_npz(model, path):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    assert model.hidden_dim == U.shape[0]
    assert model.word_dim == U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print("Loaded model parameters from {0}. hidden_dim={1} word_dim={2}".format(path, U.shape[0], U.shape[1]))

'''
    Generate sentences
'''
def gen_sentences(model, word2index, index2word, num_sent, sent_min_len):

    def gen_sentence():
        # sentence starts with START token
        sentence = [ word2index[token_start] ]
        # keep generating words until we get an END token
        while not sentence[-1] == word2index[token_end]:
            # get probability of next word
            p_next_word = model.forward(sentence)
            # Sampling :
            #   sampled_word
            sampled_word = word2index[token_unknown]
            # keep going until we get a word that isn't UNK token
            while sampled_word == word2index[token_unknown]:
                # multinomial
                #  sort : np.sort(op[-1])[-10:]
                # sample = np.random.multinomial(1, np.sort(p_next_word[-1])[-10:])
                sampled_word = np.random.choice(np.argsort(p_next_word[-1])[-10:-2])
                # sample word
                # sampled_word = sample.argmax()
            #print('Next word : {}'.format(sampled_word))
            # now that we have a word that isn't UNK
            #   append to sentence (response)
            sentence.append(sampled_word)
            #print('Sentence : {}'.format(decode_sentence(sentence,index2word,word2index)))
            if sentence[-1] == word2index[token_start] or len(sentence) > 30 or sentence[-1] == 2:
                break
        # finally, decode list of indices to a sentence
        return sentence

    sentences = []
    for i in range(num_sent):
        sentence = []
        while len(sentence) < sent_min_len:
            # keep generating new sentences
            sentence = gen_sentence()
            dec_sentence = decode_sentence(sentence, index2word, word2index)
            print('\nGen : {}'.format(dec_sentence))
        # append the sentence (len > sent_min_len)
        sentences.append(dec_sentence)

    return sentences




















