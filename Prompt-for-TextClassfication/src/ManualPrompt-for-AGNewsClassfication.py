# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 23/08/2021 13:19
@Author: yao
"""

import os
import logging
from collections import defaultdict

import transformers
import numpy as np
# transformers.datasets
from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForMaskedLM

# os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '4'
transformers.set_seed(1)
logging.basicConfig(level=logging.DEBUG)


# 1. Data Preprocessing.
"""
Class Index,Title,Description
3,Wall St. Bears Claw Back Into the Black (Reuters),"Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."

AG's News 新闻分类任务集，给定抬头headline a 和文本 text body b，
需要分类 World， Sports，Business， Science/Tech几个类别，
对于给定的 x=(a, b)，定义模板
P1(x) = -----: a b
P2(x) = a ( ----- ) b
P3(x) = ----- – a b
P4(x) = a b (---- )
P5(x) = ---- News: a b
P6(x) = [Category: ----] a b
"""

idx_to_label = {
    '1': 'world',
    '2': 'sports',
    '3': 'business',
    '4': 'science',

}

# 选用 P5(x)
class LecCallTag:

    @staticmethod
    def data_statistic(data_file):

        tags_data_dict = defaultdict(set)
        tags_count_dict = defaultdict(int)
        with open(data_file) as f:
            f.readline()
            for line in f:
                label_idx, title, description = line.strip().split('\t')

                tags_data_dict[label_idx].add(f'{title} {description}')
                tags_count_dict[label_idx] += 1
        logging.info(f'Class Count: {tags_count_dict}')
        logging.info(f'Data Size: {sum(tags_count_dict.values()):.2f}')
        return tags_data_dict

    @staticmethod
    def data_process(data_file):
        _text = []
        _label = []
        with open(data_file) as f:
            f.readline()
            for line in f:
                label_idx, title, description = line.strip().split('\t')

                # P5(x) = ---- News: a b
                _text.append(f'[MASK] News: {title} {description}')
                _label.append(f'{idx_to_label[label_idx]} News: {title} {description}')

        return _text, _label

    @staticmethod
    def create_model_tokenizer(model_name):
        _tokenizer = BertTokenizer.from_pretrained(model_name)
        _model = BertForMaskedLM.from_pretrained(model_name)
        return _tokenizer, _model

    @staticmethod
    def create_dataset(_text, _label, _tokenizer, _max_len):
        X_train, X_test, Y_train, Y_test = train_test_split(_text, _label, test_size=0.2, random_state=1)
        # logging.INFO(f'Train Size: {len(X_train)}')
        logging.info(f'Train Size: {len(X_train):.2f}, '
                     f'Test Size: {len(X_test):.2f}')

        train_dict = {'text': X_train, 'label_text': Y_train}
        test_dict = {'text': X_test, 'label_text': Y_test}

        _train_dataset = Dataset.from_dict(train_dict)
        _test_dataset = Dataset.from_dict(test_dict)

        def preprocess_function(examples):
            text_token = _tokenizer(examples[ 'text' ], padding=True,
                                    truncation=True, max_length=_max_len)
            text_token['labels'] = np.array(
                _tokenizer(examples[ 'label_text' ], padding=True,
                           truncation=True, max_length=_max_len)[ "input_ids" ]
            )
            return text_token

        _train_dataset = _train_dataset.map(preprocess_function, batched=True)
        _test_dataset = _test_dataset.map(preprocess_function, batched=True)

        return _train_dataset, _test_dataset

    @staticmethod
    def create_trainer(_model, _train_dataset, _test_dataset,
                       _checkpoint_dir, _batch_size):
        if not os.path.exists(_checkpoint_dir):
            os.mkdir(_checkpoint_dir)

        args = TrainingArguments(
            _checkpoint_dir,
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=_batch_size,
            per_device_eval_batch_size=_batch_size,
            num_train_epochs=15,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        def compute_metrics(pred):
            # 0-dim means prompt dimension
            labels = pred.label_ids[:, 0]
            preds = pred.predictions[:, 0].argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        _trainer = Trainer(
            _model,
            args,
            train_dataset=_train_dataset,
            eval_dataset=_test_dataset,
            compute_metrics=compute_metrics
        )
        return _trainer

# Running.
if __name__ == '__main__':

    train_data_path = '../data/AGNews/train.txt'
    test_data_path = '../data/AGNews/test.txt'
    checkpoint_dir = '../checkpoint'

    batch_size = 16
    max_len = 100

    lct = LecCallTag()
    # only train_data used.
    tags_data = lct.data_statistic(train_data_path)
    text, label = lct.data_process(train_data_path)
    tokenizer, model = lct.create_model_tokenizer('bert-base-cased')
    train_dataset, test_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    trainer = lct.create_trainer(model, train_dataset, test_dataset, checkpoint_dir, batch_size)
    trainer.train()
