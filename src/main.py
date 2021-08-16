# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 15/08/2021 21:09
@Author: yao
"""
import notebook.nbextensions

"""
This code is used for pytorch to implement the translation
model in order to train the generative model building.

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

"""
[KEY: > input, = target, < output]

> il est en train de peindre un tableau .
= he is painting a picture .
< he is painting a picture .

> pourquoi ne pas essayer ce vin delicieux ?
= why not try that delicious wine ?
< why not try that delicious wine ?

> elle n est pas poete mais romanciere .
= she is not a poet but a novelist .
< she not not a poet but a novelist .

> vous etes trop maigre .
= you re too skinny .
< you re all alone .
"""

import unicodedata
import string
import re
import random
import logging

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. data pre-processing

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    logging.info("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    logging.info("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    logging.info("Trimmed to %s sentence pairs" % len(pairs))
    logging.info("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    logging.info("Counted words:")
    logging.info(input_lang.name, input_lang.n_words)
    logging.info(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
logging.info(random.choice(pairs))


# 2. model building

# 2.1 Encoder (RNN)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # setting: input_size == vocab_size
        # setting: embedding_size == hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 2.2 Decoder(RNN)
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 2.3 Attention + RNN Decoder]
"""
To train we run the input sentence through the
 encoder, and keep track of every output and the 
 latest hidden state. 

Then the decoder is given the <SOS> token as its 
first input, and the last hidden state of the 
encoder as its first hidden state.

To train we run the input sentence through the
encoder, and keep track of every output and
the latest hidden state. 


You can observe outputs of teacher-forced networks
that read with coherent grammar but wander far from 
the correct translation - intuitively it has learned
to represent the output grammar and can “pick up” the
meaning once the teacher tells it the first few words, 
but it has not properly learned how to create the 
sentence from the translation in the first place.

Because of the freedom PyTorch’s autograd gives us, 
we can randomly choose to use teacher forcing or not 
with a simple if statement. Turn teacher_forcing_ratio 
up to use more of it.
"""

class AttnDecoderRNN(nn.Module):
    # embedding_size, output_size(word_size of output_lang)
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # embedded为当前时刻的输入（teacher forcing为正确target word，否则为predicted word）
        # hidden 为上一时刻的hidden state
        # 将他们链接在一起表示当前时刻的
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 3. training data preparing
# 从句子转换为index
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# 句子转换为index的tensor
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# 同时将input和target转换为index的tensor
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# 4. Training
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # 初始化第一个hidden state, 对应上一时刻hidden state的输出
    # torch.zeros(1, 1, self.hidden_size, device=device)
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 用于储存每一时刻的encoder输出
    # max_length, hidden_size
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # 遍历输入句子中的每一个单词 通过encoder进行编码
    # 储存每一时刻encoder的输出（encoder_outputs）和最后时刻输出的hidden_state
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        # 当前时刻encoder的输出 encoder_output [1, 1, hidden_state]
        # 当前时刻输出的hidden_state [1, 1, hidden_state]
        logging.debug(f'step: {ei}, encoder_output: {encoder_output.shape}, encoder_hidden: {encoder_hidden.shape}')
        encoder_outputs[ei] = encoder_output[0, 0]
        # 等同于encoder_output.squeeze_(0).squeeze_(0)
        # [hidden_state] 填入encoder_output保存encoder每一时刻的输出
        logging.debug(f'encoder_output[0, 0]: {encoder_output[ 0, 0 ].shape}')

    # 第一时刻的decoder_input为[SOS_token]对应的index
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # decoder_hidden为encoder最后一时刻的hidden state
    # 1, 1, hidden_size
    decoder_hidden = encoder_hidden
    logging.debug(f'decoder_hidden: {decoder_hidden.shape}')

    # 是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        # 全部时刻的encoder_output用于计算attention
        for di in range(target_length):
            # decoder_input 为当前时刻的输入单词 起始为[SOS]，之后为前一时刻单词（预测/真实）[1,1]
            # decoder_hidden 为上一时刻的hidden_state， 起始为encoder的最后输出，之后为上一时刻decoder的hidden_state [1, 1, hidden_size]
            # encoder_output 为encoder每一时刻的输出 大于input_length的部分为0 [max_length, hidden_size]
            logging.debug(f'decoder_input: {decoder_input.shape}, decoder_hidden: {decoder_hidden.shape},'
                          f'encoder_outputs: {encoder_outputs.shape}')
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output 为当前时刻decoder的输出 通过log_softmax，维度为 [1, vocab_size(output_lang)]
            # decoder_hidden 为当前时刻输出的hidden_state [1, 1, hidden_state]
            # decoder_attention 为计算得到的当前时刻的attention_wight [1, max_length]
            logging.debug(f'decoder_output: {decoder_output.shape}, decoder_hidden: {decoder_hidden.shape},'
                          f'decoder_attention: {decoder_attention.shape}')
            # 计算交叉熵损失
            loss += criterion(decoder_output, target_tensor[di])
            # 因为是teacher_forcing, 所以用target的单词作为下一时刻的decoder_input [1, 1]
            decoder_input = target_tensor[di]  # Teacher forcing
            # input()

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # decoder_input 为当前时刻的输入单词 起始为[SOS]，之后为前一时刻单词（预测/真实）[1,1]
            # decoder_hidden 为上一时刻的hidden_state， 起始为encoder的最后输出，之后为上一时刻decoder的hidden_state [1, 1, hidden_size]
            # encoder_output 为encoder每一时刻的输出 大于input_length的部分为0 [max_length, hidden_size]
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output 为当前时刻decoder的输出 通过log_softmax，维度为 [1, vocab_size(output_lang)]
            # decoder_hidden 为当前时刻输出的hidden_state [1, 1, hidden_state]
            # decoder_attention 为计算得到的当前时刻的attention_wight [1, max_length]
            topv, topi = decoder_output.topk(1)
            # 求tensor中某个dim的前k大或者前k小的值以及对应的index
            # 其中topv为最大的值，topi为对应的index，及预测的单词
            logging.debug(f'topv: {topv.shape}, topi: {topi.shape}')
            decoder_input = topi.squeeze().detach()  # detach from history as input

            # 遍历完输出长度，或者预测为终止标签 则结束
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# train process visualization
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# 估计剩余训练时间
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# Train
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Non-batch training
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        # single data
        # sentence_length, 1
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # 放入train函数对单条数据计算loss
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logging.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output,
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.sequeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


hidden_size = 256
# word_size(input_lang), hidden_size(embedding_size)
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# embedding_size, word_size
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    logging.info('input =', input_sentence)
    logging.info('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")


"""
Try with a different dataset
Another language pair
Human → Machine (e.g. IOT commands)
Chat → Response
Question → Answer
Replace the embeddings with pre-trained word embeddings such as word2vec or GloVe
Try with more layers, more hidden units, and more sentences. Compare the training time and results.
If you use a translation file where pairs have two of the same phrase (I am test \t I am test), you can use this as an autoencoder. Try this:
Train as an autoencoder
Save only the Encoder network
Train a new Decoder for translation from there
Total running time of the script: ( 21 minutes 41.623 seconds)
"""