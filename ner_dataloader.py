# -*- coding: utf-8 -*-

""" DataLoader
"""

import os
import re
import json
import random
import pickle
from collections import defaultdict

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from langml.utils import pad_sequences

from model import VariationalAutoencoder, Autoencoder


# set random seed
seed_value = int(os.getenv('RANDOM_SEED', -1))
if seed_value != -1:
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = string.replace("\n", "")
    string = string.replace("\t", "")
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class DataGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def set_dataset(self, train_set):
        self.train_set = train_set
        self.train_size = len(self.train_set)
        self.train_steps = len(self.train_set) // self.batch_size
        if self.train_size % self.batch_size != 0:
            self.train_steps += 1

    def __iter__(self, shuffle=True):
        while True:
            idxs = list(range(self.train_size))
            if shuffle:
                np.random.shuffle(idxs)
            batch_token_ids, batch_segment_ids, batch_tfidf, batch_label_ids = [], [], [], []
            for idx in idxs:
                d = self.train_set[idx]
                batch_token_ids.append(d['token_ids'])
                batch_segment_ids.append(d['segment_ids'])
                batch_tfidf.append(d['tfidf_vector'])
                batch_label_ids.append(d['label_id'])
                if len(batch_token_ids) == self.batch_size or idx == idxs[-1]:
                    batch_token_ids = pad_sequences(batch_token_ids, padding='post', truncating='post')
                    batch_segment_ids = pad_sequences(batch_segment_ids, padding='post', truncating='post')
                    batch_tfidf = np.array(batch_tfidf)
                    batch_label_ids = pad_sequences(batch_label_ids, padding='post', truncating='post')
                    yield [batch_token_ids, batch_segment_ids, batch_tfidf], batch_label_ids
                    batch_token_ids, batch_segment_ids, batch_tfidf, batch_label_ids = [], [], [], []

    @property
    def steps_per_epoch(self):
        return self.train_steps


class DataLoader:
    def __init__(self, dataset_name, tokenizer, max_len=512, ae_latent_dim=128, use_vae=False, batch_size=64, ae_epochs=20):
        self._train_set = []
        self._dev_set = []
        self._test_set = []

        self.dataset_name = dataset_name
        self.use_vae = use_vae
        self.batch_size = batch_size
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.train_steps = 0
        self.tokenizer = tokenizer
        self._label_size = None
        self.max_len = max_len

        self.pad = '<pad>'
        self.unk = '<unk>'
        self.tfidf = TfidfVectorizer(stop_words='english', min_df=3, max_features=5000)
        self.autoencoder = None

    def init_autoencoder(self):
        if self.autoencoder is None:
            if self.use_vae:
                print('>>> self.ae_latent_dim:', self.ae_latent_dim)
                self.autoencoder = VariationalAutoencoder(
                    latent_dim=self.ae_latent_dim, epochs=self.ae_epochs, batch_size=self.batch_size)
            else:
                self.autoencoder = Autoencoder(latent_dim=self.ae_latent_dim, epochs=self.ae_epochs, batch_size=self.batch_size)
            self.autoencoder._compile(len(self.tfidf.vocabulary_))

    def load_vocab(self, save_path):
        with open(save_path, 'rb') as reader:
            obj = pickle.load(reader)
            for key, val in obj.items():
                setattr(self, key, val)

    def save_autoencoder(self, save_path):
        self.autoencoder.autoencoder.save_weights(save_path)

    def load_autoencoder(self, save_path):
        self.init_autoencoder()
        self.autoencoder.autoencoder.load_weights(save_path)

    def set_train(self):
        """set train dataset"""
        self._train_set = self._read_data(self.dataset_name, "train", is_train=True)

    def set_dev(self):
        """set dev dataset"""
        self._dev_set = self._read_data(self.dataset_name, "validation")

    def set_test(self):
        """set test dataset"""
        self._test_set = self._read_data(self.dataset_name, "test")

    @property
    def train_set(self):
        return self._train_set

    @property
    def dev_set(self):
        return self._dev_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def label_size(self):
        return self._label_size

    def prepare_tfidf(self, data, is_train=False):
        if self.use_vae:
            print("batch alignment...")
            print("previous data size:", len(data))
            keep_size = len(data) // self.batch_size
            data = data[:keep_size * self.batch_size]
            print("alignment data size:", len(data))
        X = self.tfidf.transform([obj['raw_text'] for obj in data]).todense()
        print('>>>tf idf vector shape:', X.shape)
        if is_train:
            self.init_autoencoder()
            self.autoencoder.fit(X)
        X = self.autoencoder.encoder.predict(X, batch_size=self.batch_size)
        print('>>> Final X shape:', X.shape)
        # decomposite
        assert len(X) == len(data)
        for x, obj in zip(X, data):
            obj['tfidf_vector'] = x.tolist()
        return data

    def _read_data(self, name, split, is_train=False):
        dataset = load_dataset(name)[split]
        data = []
        tfidf_corpus = []
        all_label_set = set()
        for obj in dataset:
            raw_text = ' '.join(obj['tokens'])
            tfidf_corpus.append(raw_text)

            first, last = None, None
            token_ids, tag_ids = [], []
            for i, (tag, token) in enumerate(zip(obj['ner_tags'], obj['tokens'])):
                all_label_set.add(tag)
                tok = self.tokenizer.encode(token)
                if i == 0:
                    first, last = tok.ids[0], tok.ids[-1]
                token_ids.extend(tok.ids[1:-1])
                tag_ids.extend([tag] * len(tok.ids[1:-1]))

            token_ids = [first] + token_ids[:self.max_len-2] + [last]
            tag_ids = [0] + tag_ids[:self.max_len-2] + [0]

            assert len(token_ids) == len(tag_ids)
            
            data.append({
                'raw_text': raw_text,
                'token_ids': token_ids,
                'segment_ids': [0] * len(token_ids),
                'label_id': tag_ids
            })

        # fit tf-idf
        
        if is_train:
            print('fit tfidf...')
            self.tfidf.fit(tfidf_corpus)
            self._label_size = len(all_label_set)
        print('start to prepare tfidf feature')
        data = self.prepare_tfidf(data, is_train=is_train)
        return data
