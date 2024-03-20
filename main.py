import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import click
from utils import *
from preprocessing import preprocessing_data
from model import SpamClf
from sklearn.metrics import classification_report


@click.command()
@click.argument("epoch", default=30)  # "epoch"
@click.argument("data_path", default="spam/raw_spam.csv")
@click.argument("model_name", default="bert-base-uncased")
def main(data_path, model_name, epoch):
    tqdm.pandas()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    preprocessed_out = preprocessing_data(data_path, model_name)
    train_dataloader = preprocessed_out["train_dataloader"]
    test_dataloader = preprocessed_out["test_dataloader"]
    torch_weights = preprocessed_out["torch_weights"]
    bert_model = preprocessed_out["bert_model"]
    test_df = preprocessed_out["test_df"]

    # define model, optimizer and loss function
    spam_clf = SpamClf(bert_model).to(device)
    optimizer = torch.optim.Adam(spam_clf.parameters(), lr=3e-7)
    loss_func = torch.nn.CrossEntropyLoss(weight=torch_weights.to(device))

    # train model
    train(
        spam_clf, train_dataloader, test_dataloader, optimizer, loss_func, device, epoch
    )

    # testing on testing set
    prediction_data = predict(spam_clf, test_dataloader, device)
    y_true = test_df.target.values

    perform_test = classification_report(y_true, prediction_data["predictions"])
    print("-" * 60)
    print(f"perform_test: ")
    print(perform_test)
    print("-" * 60)


if __name__ == "__main__":
    main()
