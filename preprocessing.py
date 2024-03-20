import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import torch
from transformers import BertModel, BertTokenizer
import nltk
import spacy
from tqdm import tqdm
from utils import *


def preprocessing_data(data_path, model_name):

    tqdm.pandas()

    df = pd.read_csv(data_path, dtype="O")

    # le = LabelEncoder()
    # target = le.fit_transform(df.Category)

    # convert label to int
    target = LabelEncoder().fit_transform(df.Category)

    # count the amount of each class
    df["target"] = target
    df.target.value_counts()

    # find all punctuation
    df["SYM"] = df.apply(lambda x: list(set(re.findall(r"[\W_]", x.Message))), axis=1)

    # parse text like Entity type, Part-of-speech
    morph_model = spacy.load("en_core_web_sm")
    stopwords = nltk.corpus.stopwords.words("english")

    df["text_norm"] = df.progress_apply(
        lambda x: normalize_txt(x.Message, morph_model, stopwords), axis=1
    )
    df["text_len"] = df.apply(lambda x: len(x.text_norm.split()), axis=1)

    # cal max token len
    # bert_local_path = 'bert-base-uncased'
    bert_model = BertModel.from_pretrained(model_name)
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    df["tokens_len"] = df.apply(
        lambda x: len(bert_tokenizer.encode(x.text_norm)), axis=1
    )
    MAX_LEN = df.tokens_len.max()

    # data split
    train_df, test_df = train_test_split(df, stratify=df.target, random_state=42)

    # print(f'train_df: {type(train_df)}')
    # print(f'train_df: {train_df.shape}')


    # raise(" stop here ")
    train_dataloader = make_dataloader(
        df=train_df,
        text_col="text_norm",
        target_col="target",
        max_len=MAX_LEN,
        tokenizer=bert_tokenizer,
        batch_size=16,
    )

    test_dataloader = make_dataloader(
        df=test_df,
        text_col="text_norm",
        target_col="target",
        max_len=MAX_LEN,
        tokenizer=bert_tokenizer,
        batch_size=16,
    )

    #  modify weights to alleviate the imbalance problem
    unique_labels = np.unique(df.target)
    class_weights = compute_class_weight("balanced", classes=unique_labels, y=df.target)
    torch_weights = torch.tensor(class_weights, dtype=torch.float32)

    result = {}
    result["df"] = df
    result["train_dataloader"] = train_dataloader
    result["test_dataloader"] = test_dataloader
    result["torch_weights"] = torch_weights
    result["bert_model"] = bert_model
    result["bert_tokenizer"] = bert_tokenizer
    result["test_df"] = test_df

    return result
    # return df, MAX_LEN, train_dataloader, test_dataloader, torch_weights
