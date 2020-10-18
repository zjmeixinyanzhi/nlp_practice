import numpy as np
import data_io as pio
import nltk

GLOVE_FILE_PATH = ".\data\glove\glove.6B.50d.txt"

class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.max_char_len = 16
        self.stride = stride
        self.charset = set()
        # self.wordset = set()
        self.embeddings_index = {}
        self.embedding_matrix = []
        self.word_list = []
        self.build_charset()
        self.load_glove(GLOVE_FILE_PATH)
        self.build_wordset()

    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        print(self.ch2id, self.id2ch)
    
    def build_wordset(self):
        idx = list(range(len(self.word_list)))
        self.w2id = dict(zip(self.word_list, idx))
        self.id2w = dict(zip(idx, self.word_list))

    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset

    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids

    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_id_char(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_id_word(self, ch):
        return self.w2id.get(ch, self.w2id['unk'])

    def get_dataset(self, ds_fp):
        ccs, qcs, cws, qws, be = [], [], [], [], []
        for _, cc, qc,cw, qw, b, e in self.get_data(ds_fp):
            ccs.append(cc)
            qcs.append(qc)

            # print(len(ccs))
            # print(map(np.array, (qc)))
            cws.append(cw)
            qws.append(qw)
            be.append((b, e))
        # print(c for c in ccs)
        return map(np.array, (ccs, qcs, cws, qws, be))

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            c_seg_list = self.seg_text(context)
            q_seg_list = self.seg_text(question)
            c_char_ids = self.get_sent_ids_char(maxlen=self.max_clen, word_list=c_seg_list)
            q_char_ids = self.get_sent_ids_char(maxlen=self.max_qlen, begin=True, word_list=q_seg_list)
            c_word_ids = self.get_sent_ids_word(maxlen=self.max_clen, word_list=c_seg_list)
            q_word_ids = self.get_sent_ids_word(maxlen=self.max_qlen, begin=True, word_list=q_seg_list)
            b, e = answer_start, answer_start + len(text)
            nb = -1
            ne = -1
            len_all_char = 0
            for i, w in enumerate(c_seg_list):
                if i == 0:
                    continue
                if b > len_all_char -1 and b <= len_all_char+len(w) -1:
                    b = i + 1
                if e > len_all_char -1 and e <= len_all_char+len(w) -1:
                    e = i + 1
                len_all_char += len(w)
                
            if ne == -1:
                b = e = 0
            yield qid, c_char_ids, q_char_ids,  c_word_ids, q_word_ids, b, e
    
        def convert2id_word(self, maxlen=None, begin=False, end=False, word_list=[]):
        ch = [ch for ch in word_list]
        ch = ['cls'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            # ch += ['sep'] * end
            ch += ['pad'] * (maxlen - len(ch))
        # else:
        #     ch += ['sep'] * end

        ids = list(map(self.get_id_word, ch))

        return ids
        
    def convert2id_char(self, max_char_len=None, maxlen=None, begin=False, end=False, word_list = []):
        char_list = []
        char_list = [[self.get_id_char('[CLS]')] + [self.get_id_char('[PAD]')] * (max_char_len-1)] * begin + char_list
        for word in word_list:
            ch = [ch for ch in word]
            if max_char_len is not None:
                ch = ch[:max_char_len]

            ids = list(map(self.get_id_char, ch))
            while len(ids) < max_char_len:
                ids.append(self.get_id_char('[PAD]'))
            char_list.append(np.array(ids))

        if maxlen is not None:
            char_list = char_list[:maxlen - 1 * end]
            # char_list += [[self.get_id_char('[SEP]')]] * end
            char_list += [[self.get_id_char('[PAD]')] * max_char_len] * (maxlen - len(char_list))
        # else:
        #     char_list += [[self.get_id_char('[SEP]')]] * end

        return char_list

    def seg_text(self, text):
        words = [word.lower() for word in nltk.word_tokenize(text)]
        return words
    
    def get_sent_ids_char(self, maxlen=0, begin=False, end=True, word_list=[]):
        return self.convert2id_char(max_char_len=self.max_char_len, maxlen=maxlen, begin=False, end=True, word_list = word_list)

    def get_sent_ids_word(self, maxlen=0, begin=False, end=True, word_list=[]):
        return self.convert2id_word(maxlen=maxlen, begin=False, end=True, word_list = self.word_list)
    
    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)

    def load_glove(self, glove_file_path):
        with open(glove_file_path, encoding='utf-8') as fr:
            for line in fr:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, sep=' ')
                self.embeddings_index[word] = coefs
                self.word_list.append(word)
                self.embedding_matrix.append(coefs)

if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
    print(p.get_id("奎"))
    