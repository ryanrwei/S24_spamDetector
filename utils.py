import numpy as np
import pandas as pd
import re
import torch
from torchmetrics import Accuracy, AUROC
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer

def normalize_txt(x, morph_model, stopwords):
    tmp_txt = re.sub(r"£[0-9]+|[0-9]+£", "NUM pounds", x)
    tmp_txt = re.sub(r"$[0-9]+|[0-9]+$", "NUM dollars", tmp_txt)
    tmp_txt = re.sub(r"%[0-9]+|[0-9]+%", "NUM percent", tmp_txt)
    tmp_txt = tmp_txt.replace("&lt;#&gt;", "")
    txt_norm = " ".join(re.sub("[\W]+", " ", tmp_txt).split()).lower()
    use_tags = {"ORG", "DATE", "GPE", "PERSON", "MONEY", "TIME", "PERCENT"}
    use_lex = {"NUM"}
    tmp_doc = morph_model(txt_norm)
    text_norm_list = []
    for token in tmp_doc:
        # Entity type: token.ent_type_
        # Part-of-speech: token.pos_
        token_txt, token_tag, token_pos = token.text, token.ent_type_, token.pos_
        if token_txt in stopwords:
            continue

        if token_tag in use_tags and token_txt != "NUM":
            token_txt = token_tag
        elif token_pos in use_lex:
            token_txt = token_pos

        text_norm_list.append(token_txt)

    txt_norm_prep = " ".join(text_norm_list)
    txt_norm_prep = re.sub(r"[0-9]+", "NUM", txt_norm_prep)

    return txt_norm_prep


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, txt, tgt, max_len, tokenizer):
        self.txt = txt
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        tmp_txt = self.txt[idx]
        tmp_tokenized = self.tokenizer.encode_plus(
            tmp_txt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tmp_tgt = self.tgt[idx]
        return {k: v.flatten() for k, v in tmp_tokenized.items()}, tmp_tgt

    def __len__(self):
        return len(self.tgt)


def make_dataloader(df, text_col, target_col, max_len, tokenizer, batch_size=1):
    tmp_dataset = CustomDataset(
        df[text_col].to_numpy(), df[target_col].to_numpy(), max_len, tokenizer
    )
    return torch.utils.data.DataLoader(
        tmp_dataset, batch_size=batch_size, num_workers=4
    )


def evaluate(model, dataloader, loss_fn, device, epoch):
    # def evaluate(model, dataloader, loss_fn, device, epoch, writer):

    e_labels = torch.tensor([], dtype=torch.long, device=device)
    e_preds = torch.tensor([], dtype=torch.long, device=device)
    e_probas = torch.tensor([], dtype=torch.float32, device=device)
    e_loss = torch.tensor([], dtype=torch.float32, device=device)

    accuracy_metric = Accuracy(task="binary").to(device)
    roc_auc_metric = AUROC(task="binary").to(device)
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            y = y.to(device)
            out = model({k: v.to(device) for k, v in X.items()})
            loss = loss_fn(out, y)

            probas = softmax(out)[:, 1]
            preds = torch.argmax(out, dim=1)

            e_labels = torch.concat([e_labels, y])
            e_preds = torch.concat([e_preds, preds])
            e_probas = torch.concat([e_probas, probas])
            e_loss = torch.concat(
                [
                    e_loss,
                    torch.tensor([loss.item()], dtype=torch.float32, device=device),
                ]
            )

            accuracy_score = accuracy_metric(e_preds, e_labels)
            roc_auc_score = roc_auc_metric(e_probas, e_labels)
            mean_loss = e_loss.mean().item()

    print(
        f"Iter: {epoch} Eval Loss: {mean_loss}  Eval Accuracy: {accuracy_score}  Eval ROC-AUC: {roc_auc_score}"
    )


def train(model, train_dataloader, test_dataloader, opt, loss_fn, device, n_epochs):

    accuracy_metric = Accuracy(task="binary").to(device)
    roc_auc_metric = AUROC(task="binary").to(device)
    softmax = torch.nn.Softmax(dim=1)

    model.train()
    for epoch in range(n_epochs):

        t_labels = torch.tensor([], dtype=torch.long, device=device)
        t_preds = torch.tensor([], dtype=torch.long, device=device)
        t_probas = torch.tensor([], dtype=torch.float32, device=device)
        t_loss = torch.tensor([], dtype=torch.float32, device=device)

        for batch in tqdm(train_dataloader):
            X, y = batch
            y = y.to(device)
            out = model(
                {
                    k: v.to(device)
                    for k, v in X.items()
                    if k in ["input_ids", "attention_mask"]
                }
            )
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            probas = softmax(out)[:, 1]
            preds = torch.argmax(out, dim=1)

            t_labels = torch.concat([t_labels, y])
            t_preds = torch.concat([t_preds, preds])
            t_probas = torch.concat([t_probas, probas])
            t_loss = torch.concat(
                [
                    t_loss,
                    torch.tensor([loss.item()], dtype=torch.float32, device=device),
                ]
            )

        accuracy_score = accuracy_metric(t_preds, t_labels)
        roc_auc_score = roc_auc_metric(t_probas, t_labels)
        mean_loss = t_loss.mean().item()

        print(
            f"Iter: {epoch} Train Loss: {mean_loss}  Train Accuracy: {accuracy_score}  Train ROC-AUC: {roc_auc_score}"
        )

        #save model
        save_dir = f"models/"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # torch.save(model.state_dict(), save_dir + 'pytorch_model.bin')
        torch.save(model.state_dict(), save_dir/'pytorch_model.bin')

        model.eval()
        evaluate(model, test_dataloader, loss_fn, device, epoch)


# Post processing
def predict(model, dataloader, device):

    e_labels = torch.tensor([], dtype=torch.long, device=device)
    e_preds = torch.tensor([], dtype=torch.long, device=device)
    e_probas = torch.tensor([], dtype=torch.float32, device=device)

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            y = y.to(device)
            out = model({k: v.to(device) for k, v in X.items()})

            probas = softmax(out)[:, 1]
            preds = torch.argmax(out, dim=1)

            e_labels = torch.concat([e_labels, y])
            e_preds = torch.concat([e_preds, preds])
            e_probas = torch.concat([e_probas, probas])

    return {
        "predictions": e_preds.detach().cpu().numpy(),
        "probas": e_probas.detach().cpu().numpy(),
    }

def tokenize_function(data, model_name):
    text = data

    device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)

    # tokens['input_ids'] = np.array(tokens['input_ids'])
    # tokens['token_type_ids'] = np.array(tokens['token_type_ids'])
    # tokens['attention_mask'] = np.array(tokens['attention_mask'])

    return tokens

    # sns.set_style("darkgrid")
    # sns.set_palette("bright")

    # sns.histplot(x = prediction_data["probas"], hue=prediction_data["predictions"], log_scale=True)
    # plt.xlabel("Probas log-scale")
    # plt.title("Probability distribution")
    # plt.show()

    # fpr, tpr, thr = roc_curve(y_true, prediction_data["probas"])
    # auc_score = roc_auc_score(y_true, prediction_data["probas"])
    # plt.plot(fpr, tpr)
    # plt.plot([0,1], [0, 1], linestyle="--")
    # plt.legend([f"ROC-CURVE\nAUC:{auc_score:.2f}", "BASE-ESTIMATOR\nAUC:0.5"])
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("ROC-AUC")
    # plt.show()