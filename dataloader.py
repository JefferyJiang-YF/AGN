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
                    batch_label_ids = np.array(batch_label_ids)
                    yield [batch_token_ids, batch_segment_ids, batch_tfidf], batch_label_ids
                    batch_token_ids, batch_segment_ids, batch_tfidf, batch_label_ids = [], [], [], []

    @property
    def steps_per_epoch(self):
        return self.train_steps


class DataLoader:
    def __init__(self, tokenizer, ae_latent_dim=128, use_vae=False, batch_size=64, ae_epochs=20):
        self._train_set = []
        self._dev_set = []
        self._test_set = []

        self.use_vae = use_vae
        self.batch_size = batch_size
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.train_steps = 0
        self.tokenizer = tokenizer
        self.label2idx = {}

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

    def save_vocab(self, save_path):
        with open(save_path, 'wb') as writer:
            pickle.dump({
                'label2idx': self.label2idx,
            }, writer)

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

    def set_train(self, train_path):
        """set train dataset"""
        self._train_set = self._read_data(train_path, build_vocab=True)

    def set_dev(self, dev_path):
        """set dev dataset"""
        self._dev_set = self._read_data(dev_path)

    def set_test(self, test_path):
        """set test dataset"""
        self._test_set = self._read_data(test_path)

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
        return len(self.label2idx)

    def save_dataset(self, setname, fpath):
        if setname == 'train':
            dataset = self.train_set
        elif setname == 'dev':
            dataset = self.dev_set
        elif setname == 'test':
            dataset = self.test_set
        else:
            raise ValueError(f'not support set {setname}')
        with open(fpath, 'w') as writer:
            for data in dataset:
                writer.writelines(json.dumps(data, ensure_ascii=False) + "\n")

    def load_dataset(self, setname, fpath):
        if setname not in ['train', 'dev', 'test']:
            raise ValueError(f'not support set {setname}')
        dataset = []
        with open(fpath, 'r') as reader:
            for line in reader:
                dataset.append(json.loads(line.strip()))
        if setname == 'train':
            self._train_set = dataset
        elif setname == 'dev':
            self._dev_set = dataset
        elif setname == 'test':
            self._test_set = dataset

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

    def _read_data(self, fpath, build_vocab=False):
        data = []
        tfidf_corpus = []
        with open(fpath, "r", encoding="utf-8") as reader:
            for line in reader:
                obj = json.loads(line)
                obj['text'] = clean_str(obj['text'])
                tfidf_corpus.append(obj['text'])
                if build_vocab:
                    if obj['label'] not in self.label2idx:
                        self.label2idx[obj['label']] = len(self.label2idx)
                tokenized = self.tokenizer.encode(obj['text'])
                token_ids, segment_ids = tokenized.ids, tokenized.segment_ids
                data.append({
                    'raw_text': obj['text'],
                    'token_ids': token_ids,
                    'segment_ids': segment_ids,
                    'label_id': self.label2idx[obj['label']]
                })

        # fit tf-idf
        
        if build_vocab:
            print('fit tfidf...')
            self.tfidf.fit(tfidf_corpus)
        print('start to prepare tfidf feature')
        data = self.prepare_tfidf(data, is_train=build_vocab)
        return data
