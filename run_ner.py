# -*- coding: utf-8 -*-

import os
import sys
import json
import random
from pprint import pprint

import numpy as np
import tensorflow as tf
from langml.tokenizer import WPTokenizer

from ner_dataloader import DataLoader, DataGenerator
from model import AGNModel
from metrics import NERMetrics


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
tokenizer = WPTokenizer(os.path.join(config['pretrained_model_dir'], 'vocab.txt'), lowercase=config.get('lowercase', False))
tokenizer.enable_truncation(max_length=config['max_len'])

print("load data...")
dataloader = DataLoader(config['dataset_name'],
                        tokenizer,
                        max_len=config['max_len'],
                        ae_latent_dim=config['ae_latent_dim'],
                        use_vae=True,
                        batch_size=config["batch_size"],
                        ae_epochs=config['ae_epochs'])
dataloader.set_train()
dataloader.set_dev()
dataloader.set_test()
dataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder.weights'))

micro_f1_list = []
for idx in range(1, config['iterations'] + 1):
    print("build generator")
    generator = DataGenerator(config['batch_size'])
    generator.set_dataset(dataloader.train_set)
    metrics_callback = NERMetrics(
        config['batch_size'],
        dataloader.test_set,
        os.path.join(config['save_dir'], f'ner_model.weights'))
    config['steps_per_epoch'] = generator.steps_per_epoch
    config['output_size'] = dataloader.label_size
    print('!!!>>>>> output size:', config['output_size'])
    model = AGNModel(config, task='ner')
    print("start to fit...")
    model.fit(
            generator.__iter__(),
            steps_per_epoch=generator.steps_per_epoch,
            epochs=config['epochs'],
            callbacks=[metrics_callback],
            verbose=config['verbose']
    )

    f1 = max(metrics_callback.history["val_f1"])
    micro_f1_list.append(f1)
    log = f"iteration {idx} f1: {f1}\n"
    print(log)

print("Average micro f1:", sum(micro_f1_list) / len(micro_f1_list))
