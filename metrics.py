# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from langml import keras
from seqeval.metrics import (
    f1_score as ner_f1_score,
    classification_report as ner_classification_report
)
from sklearn.metrics import f1_score, accuracy_score
from boltons.iterutils import chunked_iter
from langml.utils import pad_sequences


class ClfMetrics(keras.callbacks.Callback):
    def __init__(self,
                 batch_size,
                 eval_data,
                 save_path,
                 min_delta=1e-4,
                 patience=10):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater

        self.save_path = save_path
        self.batch_size = batch_size
        self.eval_data = eval_data
        self.history = defaultdict(list)

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def calc_metrics(self):
        y_true, y_pred = [], []
        for chunk in chunked_iter(self.eval_data, self.batch_size):
            token_ids = [obj['token_ids'] for obj in chunk]
            segment_ids = [obj['segment_ids'] for obj in chunk]
            tfidf_vectors = [obj['tfidf_vector'] for obj in chunk]
            true_labels = [obj['label_id'] for obj in chunk]

            token_ids = pad_sequences(token_ids, padding='post', truncating='post')
            segment_ids = pad_sequences(segment_ids, padding='post', truncating='post')
            tfidf_vectors = np.array(tfidf_vectors)
            pred = self.model([token_ids, segment_ids, tfidf_vectors])
            pred = np.argmax(pred, 1)
            y_true += list(true_labels)
            y_pred += list(pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1, acc

    def on_epoch_end(self, epoch, logs=None):
        val_f1, val_acc = self.calc_metrics()
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
        print(f"- val_acc {val_acc} - val_f1 {val_f1}")
        if self.monitor_op(val_f1 - self.min_delta, self.best) or self.monitor_op(self.min_delta, val_f1):
            self.best = val_f1
            self.wait = 0
            print(f'new best model, save model to  {self.save_path}...')
            self.model.save_weights(self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


class NERMetrics(keras.callbacks.Callback):
    def __init__(self,
                 batch_size,
                 eval_data,
                 save_path,
                 min_delta=1e-4,
                 patience=10):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater

        self.save_path = save_path
        self.batch_size = batch_size
        self.eval_data = eval_data
        self.history = defaultdict(list)

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def decode(self, tag_list):
        ret = []
        prev = None
        for tag_id in tag_list:
            if tag_id in [0, 101, 102]:
                ret.append('O')
                prev = None
                continue
            if prev is not None:
                if prev == tag_id:
                    ret.append(f'I-{tag_id}')
                else:
                    ret.append(f'B-{tag_id}')
            else:
                ret.append(f'B-{tag_id}')
            prev = tag_id
        return ret

    def calc_metrics(self):
        y_true, y_pred = [], []
        for chunk in chunked_iter(self.eval_data, self.batch_size):
            token_ids = [obj['token_ids'] for obj in chunk]
            segment_ids = [obj['segment_ids'] for obj in chunk]
            tfidf_vectors = [obj['tfidf_vector'] for obj in chunk]
            true_labels = [obj['label_id'] for obj in chunk]

            token_ids = pad_sequences(token_ids, padding='post', truncating='post')
            segment_ids = pad_sequences(segment_ids, padding='post', truncating='post')
            tfidf_vectors = np.array(tfidf_vectors)
            pred = self.model([token_ids, segment_ids, tfidf_vectors])
            pred = np.argmax(pred, -1)
            size = pred[0].shape[-1]
            padding_true_labels = []
            for label in true_labels:
                padding_true_labels.append(label + [0] * (size - len(label)))
            y_true += [self.decode(t) for t in padding_true_labels]
            y_pred += [self.decode(t) for t in pred]
        # print("y_true>>>", y_true)
        # print("y_pred>>>", y_pred)
        print(ner_classification_report(y_true, y_pred))
        f1 = ner_f1_score(y_true, y_pred, average="micro")
        return f1

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = self.calc_metrics()
        self.history['val_f1'].append(val_f1)
        print(f"- val_f1 {val_f1}")
        if self.monitor_op(val_f1 - self.min_delta, self.best) or self.monitor_op(self.min_delta, val_f1):
            self.best = val_f1
            self.wait = 0
            print(f'new best model, save model to  {self.save_path}...')
            self.model.save_weights(self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
