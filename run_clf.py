# -*- coding: utf-8 -*-

import os
import sys
import json
import random
from pprint import pprint

import numpy as np
import tensorflow as tf
from langml.tokenizer import WPTokenizer

from dataloader import DataLoader, DataGenerator
from model import AGNModel
from metrics import ClfMetrics


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Failed to set gpu memory growth!')

# set random seed
seed_value = int(os.getenv('RANDOM_SEED', -1))
if seed_value != -1:
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


if len(sys.argv) != 2:
    print("usage: python main.py /path/to/config")
    exit()

config_file = str(sys.argv[1])

with open(config_file, "r") as reader:
    config = json.load(reader)

print("config:")
pprint(config)

# create save_dir folder if not exists
if not os.path.exists(config['save_dir']):
    os.makedirs(config['save_dir'])


# Load tokenizer
tokenizer = WPTokenizer(os.path.join(config['pretrained_model_dir'], 'vocab.txt'), lowercase=True)
tokenizer.enable_truncation(max_length=config['max_len'])

print("load data...")
dataloader = DataLoader(tokenizer,
                        config['ae_latent_dim'],
                        use_vae=True,
                        batch_size=config["batch_size"],
                        ae_epochs=config['ae_epochs'])
dataloader.set_train(config['train_path'])
dataloader.set_dev(config['dev_path'])
dataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder.weights'))
dataloader.save_vocab(os.path.join(config['save_dir'], 'vocab.pickle'))

accuracy_list = []
f1_list = []
for idx in range(1, config['iterations'] + 1):
    print("build generator")
    generator = DataGenerator(config['batch_size'])
    generator.set_dataset(dataloader.train_set)
    metrics_callback = ClfMetrics(
        config['batch_size'],
        dataloader.dev_set,
        os.path.join(config['save_dir'], 'clf_model.weights'))
    config['steps_per_epoch'] = generator.steps_per_epoch
    config['output_size'] = dataloader.label_size
    model = AGNModel(config)
    print("start to fit...")
    model.model.fit(
            generator.__iter__(),
            steps_per_epoch=generator.steps_per_epoch,
            epochs=config['epochs'],
            callbacks=[metrics_callback],
            verbose=config['verbose']
    )

    accuracy = max(metrics_callback.history["val_acc"])
    f1 = max(metrics_callback.history["val_f1"])
    accuracy_list.append(accuracy)
    f1_list.append(f1)
    log = f"iteration {idx} accuracy: {accuracy}, f1: {f1}\n"
    print(log)

print("Average accuracy:", sum(accuracy_list) / len(accuracy_list))
print("Average f1:", sum(f1_list) / len(f1_list))
