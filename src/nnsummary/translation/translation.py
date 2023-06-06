import ctranslate2
import sentencepiece as spm
import json
from tqdm import tqdm
import pandas as pd
import hashlib
import pickle
import os
import nltk

from nnsummary.config import PATH_TRANSLATION_EN2NL, CACHE_DIR, PATH_TRANSLATION_NL2EN, DATA_DIR


class Translator:

    def __init__(self):
        self.nlen_sp = spm.SentencePieceProcessor()
        self.nlen_sp.load(os.path.join(PATH_TRANSLATION_NL2EN, 'source.spm'))
        self.nlen_translator = ctranslate2.Translator(os.path.join(DATA_DIR, "nlen_ctranslate2"))
        self.nlen_cache = {}
        self.nlen_cache_path = os.path.join(CACHE_DIR, 'nlen_cache.pkl')

        self.ennl_sp = spm.SentencePieceProcessor()
        self.ennl_sp.load(os.path.join(PATH_TRANSLATION_EN2NL, 'source.spm'))
        self.ennl_translator = ctranslate2.Translator(os.path.join(DATA_DIR, "ennl_ctranslate2"))
        self.ennl_cache = {}
        self.ennl_cache_path = os.path.join(CACHE_DIR, 'ennl_cache.pkl')

        if os.path.isfile(self.nlen_cache_path) and os.path.isfile(self.ennl_cache_path):
            self.load_cache()

    def commit_cache(self):
        print(f'Write cache to files {self.nlen_cache_path} and {self.ennl_cache_path}')
        with open(self.nlen_cache_path, 'wb') as f:
            pickle.dump(self.nlen_cache, f)

        with open(self.ennl_cache_path, 'wb') as f:
            pickle.dump(self.ennl_cache, f)
        print('Finished')

    def load_cache(self):
        print(f'Load cache from files {self.nlen_cache_path} and {self.ennl_cache_path}')
        if os.path.isfile(self.nlen_cache_path) and os.path.isfile(self.ennl_cache_path):
            with open(self.nlen_cache_path, 'rb') as f:
                self.nlen_cache = pickle.load(f)

            with open(self.ennl_cache_path, 'rb') as f:
                self.ennl_cache = pickle.load(f)
            print('Finished')
        else:
            print('There is no cache yet - Nothing happened')

    def translate_nl_to_en(self, text):
        trans_sentences = []
        for s in nltk.sent_tokenize(text):
            s = s.strip()
            md5 = hashlib.md5(s.encode()).hexdigest()

            if md5 in self.nlen_cache:
                trans_sentences.append(self.nlen_cache[md5][1])
                continue

            source = self.nlen_sp.encode(s, out_type=str)
            results = self.nlen_translator.translate_batch([source])
            translated_s = self.nlen_sp.decode(results[0].hypotheses[0]).replace('▁', ' ')
            self.nlen_cache[md5] = [s, translated_s]
            trans_sentences.append(translated_s)

        return ' '.join([s for s in trans_sentences])

    def translate_en_to_nl(self, text):
        trans_sentences = []
        for s in nltk.sent_tokenize(text):
            s = s.strip()
            md5 = hashlib.md5(s.encode()).hexdigest()

            if md5 in self.ennl_cache:
                trans_sentences.append(self.ennl_cache[md5][1])
                continue

            source = self.ennl_sp.encode(s, out_type=str)
            results = self.ennl_translator.translate_batch([source])
            translated_s = self.ennl_sp.decode(results[0].hypotheses[0]).replace('▁', ' ')
            self.ennl_cache[md5] = [s, translated_s]
            trans_sentences.append(translated_s)
        return ' '.join([s for s in trans_sentences])
