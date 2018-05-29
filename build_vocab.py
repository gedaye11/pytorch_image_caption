import re
import pickle
from collections import Counter
import nltk


class Vocab:
    '''vocabulary'''
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.ix = 0

    def add_word(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.ix
            self.i2w[self.ix] = word
            self.ix += 1

    def __call__(self, word):
        if word not in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[word]

    def __len__(self):
        return len(self.w2i)


def build_vocab(mode_list=['factual', 'humorous']):
    '''build vocabulary'''
    # define vocabulary
    vocab = Vocab()
    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # add words
    for mode in mode_list:
        if mode == 'factual':
            captions = extract_captions(mode=mode)
            words = nltk.tokenize.word_tokenize(captions)
            counter = Counter(words)
            words = [word for word, cnt in counter.items() if cnt >= 2]
        else:
            captions = extract_captions(mode=mode)
            words = nltk.tokenize.word_tokenize(captions)

        for word in words:
            vocab.add_word(word)

    return vocab


def extract_captions(mode='factual'):
    '''extract captions from data files for building vocabulary'''
    text = ''
    if mode == 'factual':
        with open("/home/gexuri/VisualSearch/AIchallengetrain_val/TextData/seg.AIchallengetrain.caption.txt", 'r') as f:
            res = f.readlines()

        for line in res:
            line = line.strip().split(' ',1)
            text += line[1] + ' '

    else:
        if mode == 'humorous':
            with open("data/seg.weibocleartrainV2.caption.txt", 'r') as f:
                res = f.readlines()
        else:
            with open("data/seg.weibocleartrainV2.caption.txt", 'r') as f:
                res = f.readlines()

        for line in res:
            line = line.strip().split(' ',1)
            text += line[1] + ' '

    return text.strip().lower()



if __name__ == '__main__':
    vocab = build_vocab(mode_list=['factual'])
    print(vocab.__len__())
    # vocab = Vocab()
    # rf = open('./data/vocab_count_thr_3.txt', 'r')
    # for line in rf:
    #     vocab.add_word(line.strip())
    # rf.close()
    print vocab
    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
