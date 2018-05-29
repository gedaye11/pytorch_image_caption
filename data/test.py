import cPickle as pickle

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


with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
print len(vocab)
print vocab.w2i['<start>']
print vocab.i2w[1887]