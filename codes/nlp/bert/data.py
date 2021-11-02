import tqdm
from typing import List, Tuple
import pprint
import torch
import numpy as np
from model import Hparam


_RawDataT = List[List[str]]   # paragraph[sentence_with_space]
_IntDataT = List[List[List[int]]]  # paragraph[sentence[int_token]]

RAW_DATA_PATH = '/home/rjia/playground/datasets/WikiText2/wikitext-2/wiki.{}.tokens'

#######################################################
# SECTION raw(txt)_data to int_data, build vocab
#######################################################

def read_raw_text(w='train') -> _RawDataT:
    """raw text -> list[list[str]], 
    - outer list is paragraph, inner list is sentence
    - to lower
    - filter so that pararaph length > 2
    """
    assert w in ('train', 'valid', 'test')
    with open(RAW_DATA_PATH.format('train')) as f:
        lines = f.readlines()
    ret = []
    for line in lines:
        sentences = line.split('.')
        sentences = [sen.strip().lower() for sen in sentences if sen.strip()]
        if len(sentences) > 2:
            ret.append(sentences)
    return ret


class Vocab:
    def __init__(self):
        self.s2i = {}
        self.i2s = {}
        self.specials = ['<unk>', '<pad>', '<cls>', '<sep>', '<msk>']
        self.UNK, self.PAD, self.CLS, self.SEP, self.MSK = 0, 1, 2, 3, 4
        self.counter = {}
        self.MIN_FREQ = 10
        self.VOCAB_SIZE = None   # set later
    
    def construct_from_paragraphs(self, raw_data: _RawDataT):
        for paragraph in tqdm.tqdm(raw_data, desc='[vocab] counting'):
            for sentence in paragraph:
                token_lst = sentence.split(' ')
                for token in token_lst:
                    token = token
                    if token in self.counter:
                        self.counter[token] += 1
                    else:
                        self.counter[token] = 0
        drop_keys = [tk for tk, n in self.counter.items() if n < self.MIN_FREQ]
        for tk in drop_keys:
            del self.counter[tk]
        self.VOCAB_SIZE = len(self.counter) + len(self.specials)
        print(f'[vocab] droped keys, VOCAB_SIZE = {self.VOCAB_SIZE}')

        i = 0
        for tk in self.specials:
            self.i2s[i] = tk
            self.s2i[tk] = i
            i += 1
        for tk in self.counter:
            if tk in self.specials:
                continue
            self.s2i[tk] = i
            self.i2s[i] = tk
            i += 1
        print('[vocab] done')
    
    def summarize(self):
        d = {
            'MIN_FREQ': self.MIN_FREQ,
            'vocab_size': len(self.i2s),
            'specials': self.specials,
            'specials_int': [self.s2i[tk] for tk in self.specials],
            'head': [(self.i2s[i], i) for i in range(10)],
        }
        pprint.pprint(d)

    def tokenlize(self, sentence: str) -> List[int]:
        return [self.s2i.get(token, self.UNK) for token in sentence.split(' ')]

    
def construct_vocab_and_get_int_data() -> Tuple[Vocab, Tuple[_IntDataT, _IntDataT, _IntDataT]]:
    train_data_raw = read_raw_text('train')
    #valid_data_raw = read_raw_text('valid')
    #test_data_raw = read_raw_text('test')
    # vocab
    vocab = Vocab()
    vocab.construct_from_paragraphs(train_data_raw)
    vocab.summarize()
    # tokenlize
    def __tokenlize(vocab: Vocab, data_raw):
        return [[vocab.tokenlize(sentence) for sentence in paragraph] for paragraph in data_raw]
    train_data = __tokenlize(vocab, train_data_raw)
    #valid_data = __tokenlize(vocab, valid_data_raw)
    #test_data = __tokenlize(vocab, test_data_raw)
    print(train_data[0])
    print(train_data[1])
    print(train_data[2])
    return vocab, (train_data, None, None)


#######################################################
# SECTION generate_batch
#######################################################

def __mask_one_sentence(sx: np.array, MSK):
    p = 0.15
    pm, pu, po = [p * p_ for p_ in [0.8, 0.1, 0.1]]
    msk_msk = np.random.choice([True, False], size=sx.shape, p=[pm, 1-pm])
    msk_unchange = np.random.choice([True, False], size=sx.shape, p=[pu, 1-pu])
    msk_other = np.random.choice([True, False], size=sx.shape, p=[po, 1-po])
    msk = msk_msk | msk_unchange | msk_other
    sy = np.copy(sx)
    sy[msk_msk] = MSK
    sy[msk_other] = np.random.choice(sx, size=sy[msk_other].shape)
    return msk, sy


def __get_two_sentence(data: _IntDataT, max_len: int):
    n_paragraph = len(data)
    g_idx = np.random.randint(0, n_paragraph)
    paragraph = data[g_idx]
    n_sentence = len(paragraph)
    if np.random.random() > 0.5:
        is_next = True
        s0_idx = np.random.randint(0, n_sentence-1)
        s1_idx = s0_idx + 1
    else:
        is_next = False
        s0_idx = np.random.randint(0, n_sentence)
        s1_idx = np.random.randint(0, n_sentence)
    s0 = np.array(paragraph[s0_idx][:max_len])
    s1 = np.array(paragraph[s1_idx][:max_len])
    return is_next, s0, s1


def get_one_sample(data: _IntDataT, max_len: int, CLS, SEP, MSK):
    CLS, SEP, FALSE, TRUE = np.array([CLS]), np.array([SEP]), np.array([False]), np.array([True])
    # two sentence
    is_next, s0, s1 = __get_two_sentence(data, max_len)
    sx = np.concatenate([CLS, s0, SEP, s1, SEP])
    # mask
    msk_s0, y_s0 = __mask_one_sentence(s0, MSK)
    msk_s1, y_s1 = __mask_one_sentence(s1, MSK)
    msk = np.concatenate([FALSE, msk_s0, FALSE, msk_s1, FALSE])
    sy = np.concatenate([CLS, y_s0, SEP, y_s1, SEP])
    # segment
    seg = np.concatenate([np.zeros(shape=(s0.size + 2, )), np.ones(shape=(s1.size + 1, ))])
    return is_next, sx, sy, msk, seg


def __stack_ragged_1dArray(la: List[np.ndarray], fill_value, dtype):
    bz, max_size = len(la), max(a.size for a in la)
    out = np.full(shape=(bz, max_size), fill_value=fill_value, dtype=dtype)
    for i, a in enumerate(la):
        out[i][:a.size] = a
    return out


def __collate(samples: List[Tuple], PAD: int):
    bz, max_seq_len = len(samples), max(tp[1].size for tp in samples)
    # (bz,)
    is_next = np.array([tp[0] for tp in samples]).astype(bool)
    # (bz, max_seq_len) -> (max_seq_len, bz)
    sx = __stack_ragged_1dArray([tp[1] for tp in samples], fill_value=PAD, dtype=np.int64).transpose()
    sy = __stack_ragged_1dArray([tp[2] for tp in samples], fill_value=PAD, dtype=np.int64).transpose()
    msk = __stack_ragged_1dArray([tp[3] for tp in samples], fill_value=False, dtype=bool).transpose()
    seg = __stack_ragged_1dArray([tp[4] for tp in samples], fill_value=1, dtype=np.int64).transpose()
    return is_next, sx, sy, msk, seg


def get_one_batch(data: _IntDataT, bz: int, max_len: int, specials: Tuple[int]):
    CLS, SEP, MSK, PAD = specials
    samples = [get_one_sample(data, max_len, CLS, SEP, MSK) for _ in range(bz)]
    batch = __collate(samples, PAD)
    return tuple(torch.from_numpy(arr) for arr in batch)


def batch_generator(data: _IntDataT, bz: int, max_len: int, specials: Tuple[int]):
    while True:
        yield get_one_batch(data, bz, max_len, specials)
    

#######################################################
# SECTION pipeline
#######################################################

class data_pipeline():
    def __init__(self, hp: Hparam) -> None:
        self.hp = hp
        # raw -> int data && vocab
        vocab, (train_data, valid_data, test_data) = construct_vocab_and_get_int_data()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.vocab = vocab
        # batch
        self.train_batch_generator = batch_generator(
            self.train_data, hp.bz, hp.max_len, 
            (vocab.CLS, vocab.SEP, vocab.MSK, vocab.PAD)
        )
        

    def test_one_sample(self):
        is_next, sx, sy, msk, seg = get_one_sample(
            self.train_data, 100, 
            self.vocab.CLS, self.vocab.SEP, self.vocab.MSK
        )
        print('--------------- one sample -----------------')
        print(is_next)
        print(sx)
        print(sy)
        print(msk)
        print(seg)
        print('--------------------------------------------')
    
    def test_batch(self):
        for step, batch in enumerate(self.train_batch_generator):
            is_next, sx, sy, msk, seg = batch
            print(is_next.shape)
            print(sx.shape)
            print(sy.shape)
            print(msk.shape)
            print(seg.shape)
            break
        print(self.train_batch_generator)



if __name__ == '__main__':
    from model import Hparam
    hp = Hparam()
    data = data_pipeline(hp)
    data.test_one_sample()
    data.test_batch()