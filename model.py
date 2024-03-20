import numpy as np
import pandas as pd
import torch
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class SpamClf(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert_model = base_model

        # fix Bert's parameter
        for param in self.bert_model.parameters():
            param.requieres_grad=False

        # only train FC layer
        self.fc1 = torch.nn.Linear(768,2)
        
    def forward(self, x):
        out = self.bert_model(**x).last_hidden_state[:, 0, :]
        out = self.fc1(out)
        return out