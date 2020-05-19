import numpy as np
import pandas as pd
import MeCab
import re
import neologdn
import string
import emoji
import torch
import torchtext
from torchtext.vocab import Vectors
import os
import random

def seed_everything(seed=1234):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(seed=1234)

# テキスト前処理
def preprocessing_text(text):
    # 英語の小文字化(表記揺れの抑制)
    text = text.lower()
    # URLの除去(neologdnの後にやるとうまくいかないかも(URL直後に文章が続くとそれも除去される)))
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', '', text)
    # tweetの前処理
    text = re.sub(r"@([A-Za-z0-9_]+) ", '', text) # リプライ
    text = re.sub(r'#(\w+)', '', text) # ハッシュタグ
    # neologdnを用いて文字表現の正規化(全角・半角の統一と重ね表現の除去)
    text = neologdn.normalize(text)
    # 数字を全て0に置換(解析タスク上、数字を重要視しない場合は語彙数増加を抑制するために任意の数字に統一したり除去することもある)
    text = re.sub(r'[0-9０-９]+', '0', text)
    # 半角記号の除去
    text = re.sub(r'[!-/:-@【】[-`{-~]', "", text)
    # 改行
    text = re.sub('\n', '', text)
    # 絵文字
    text = ''.join(['' if c in emoji.UNICODE_EMOJI else c for c in text])
    # 中黒や三点リーダ
    text = re.sub(r'[・…]', '', text)
    return text

# MeCab + NEologdによるtokenizer
def tokenizer_mecab(text):
    tagger = MeCab.Tagger('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd') # -Owakatiで分かち書きのみ出力
    text = tagger.parse(text)
    text = text.strip().split()
    return text

# pipeline
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_mecab(text)
    
    return ret

def get_dataset(max_length=256, split_ratio=[0.92, 0.04, 0.04]):
    PATH = '/content/drive/My Drive/Colab Notebooks/NLP/RionTweetClassifier/data/rion_corpus.csv'
    # Field
    TEXT = torchtext.data.Field(sequential=True, 
                                tokenize=tokenizer_with_preprocessing, 
                                use_vocab=True, 
                                lower=True, 
                                include_lengths=True, 
                                batch_first=True, 
                                fix_length=max_length, 
                                init_token="<cls>", 
                                eos_token="<eos>"
                                )

    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float32)
    # Dataset
    ds = torchtext.data.TabularDataset(path=PATH, 
                                       format='csv', 
                                       skip_header=True, 
                                       fields=[('Text', TEXT), ('Label', LABEL)]
                                       )
    
    train_ds, test_ds, val_ds = ds.split(split_ratio=split_ratio)
    # embedding
    FASTTEXT = '/content/drive/My Drive/Colab Notebooks/NLP/nlp_tutorial/model.vec'
    fastText_vectors = Vectors(name=FASTTEXT)
    # vocab
    TEXT.build_vocab(train_ds, vectors=fastText_vectors, min_freq=1)

    return train_ds, test_ds, val_ds, TEXT