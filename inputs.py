from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from functools import reduce

from nltk.tokenize import word_tokenize
import codecs
import tensorflow as tf
import re
import collections
import numpy as np

BATCH_SIZE = 33
VOCAB_SIZE = 18592
SEQUENCE_LENGTH = 37
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH = 10662
NUM_EXPOCHES = 2

def clean_sentence(sentence):
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower()

'''
建立词典
'''
def build_vocab():
    positive_sentences = codecs.open('test.pos').readlines()
    negative_sentences = codecs.open('test.neg').readlines()
    num_positive = len(positive_sentences)
    sentences = positive_sentences + negative_sentences
    clean = map(lambda sentence: word_tokenize(clean_sentence(sentence)), sentences)
    line = reduce(lambda x,y: x+y, clean)
    counter = collections.Counter(line)
    count_pairs = sorted(counter.items(),key=lambda x:(-x[1],x[0]))
    words, _ = list(zip(*count_pairs))
    word2id = dict(zip(words, range(2, len(words)+2)))
    word2id['<pad>'] = -1
    word2id['<sos>'] = 0
    word2id['<eos>'] = 1
    vocab = list(words) + ['<pad>', '<sos>', '<eos>']
    array = [[0] + [word2id[word] for word in sent] + [1] for sent in clean] # what's this? i don't know!
    return vocab, word2id, array, num_positive

def input_producer(train=True):
    vocab, word2id, array, num_positive = build_vocab()
    num_sents = len(array)
    labels = np.ones([num_sents])
    labels[num_positive+1:] = 0
    max_length = max(map(len,array))
    pad = map(lambda sent: sent + [-1] * (max_length - len(sent)), array)
    data = np.hstack((np.array(pad), np.expand_dims(labels,1)))
    np.random.shuffle(data)

    '''
    训练集合标签转化为tensor
    '''
    total_inputs = tf.convert_to_tensor(data[:,:-1])
    total_inputs = tf.cast(x=total_inputs, dtype=tf.int32)

    total_labels = tf.convert_to_tensor(data[:,-1])
    total_labels = tf.cast(x=total_labels, dtype=tf.int64)

    i = tf.train.range_input_producer(NUM_EXPOCHES, shuffle=False).dequeue()
    if train:
         inputs = tf.slice(total_inputs, [i * BATCH_SIZE,0], [BATCH_SIZE, max_length])
         labels = tf.slice(total_labels, [i * BATCH_SIZE], [BATCH_SIZE])
    else:
        inputs = total_inputs
        labels = total_labels
    return inputs, labels

class Inputs(object):
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.inputs, self.labels = input_producer()
        self.vocab_size = VOCAB_SIZE
        self.sequence_length = SEQUENCE_LENGTH
        self.num_classes = NUM_CLASSES



