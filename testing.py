import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_score, f1_score
import os
from datasets import load_dataset, DatasetDict, Dataset

# from torch.utils.data import Dataset, DataLoader #Data
from transformers import EarlyStoppingCallback, IntervalStrategy

import numpy as np
from tqdm import tqdm

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from model import SpamClf
from transformers import BertModel, BertTokenizer
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load trained model
save_dir = f"models/"
model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
spam_clf = SpamClf(bert_model)
spam_clf.load_state_dict(torch.load(f"{save_dir}/pytorch_model.bin"))
spam_clf.eval()

# ask user for msg
input_msg = input(f"please send the message: \n")

# tokenization
msg_tokens = tokenize_function([input_msg], model_name)

# input msg to model
preds = spam_clf(msg_tokens)
preds = torch.argmax(preds, dim=1)
output = "ham" if preds == 0 else "spam"
print(f"output: {output}")