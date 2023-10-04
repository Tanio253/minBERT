import torch
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score
from types import Dict
def MultiEvaluation(model, sst_dl, quora_dl, sts_dl):
    model.eval()
    y_true_sst = []
    y_pred_sst = []
    sents_sst = []
    sent_ids_sst = []
    with torch.no_grad():
        for batch in tqdm(sst_dl, desc = f'SST Evaluation'):
            input_ids, attention_mask, labels, sent, sent_id = batch
            logits = model.predict_sentiment(input_ids, attention_mask)
            y_pred = torch.argmax(logits, dim = 1).cpu().numpy().flatten()
            y_true = labels.cpu().numpy().flatten()
            y_true_sst.extend(y_true)
            y_pred_sst.extend(y_pred)
            sents_sst.extend(sent)
            sent_ids_sst.extend(sent_id)
        sst_accuracy = accuracy_score(y_true_sst,y_pred_sst)
    y_true_quora = []
    y_pred_quora = []
    sents_quora1 = []
    sent_ids_quora = []
    sents_quora2 = []
    with torch.no_grad():
        for batch in tqdm(quora_dl, desc = f'Quora Evaluation'):
            input_ids, attention_mask, labels, sent, sent_id = batch
            input_ids1, input_ids2 = input_ids
            attention_mask1, attention_mask2 = attention_mask
            sent1, sent2 = sent
            logits = model.predict_paraphrase(input_ids1, attention_mask1, input_ids2, attention_mask2)
            y_pred = logits.cpu().numpy().flatten()
            y_true = labels.cpu().numpy().flatten()
            y_true_quora.extend(y_true)
            y_pred_quora.extend(y_pred)
            sents_quora1.extend(sent1)
            sents_quora2.extend(sent2)
            sent_ids_quora.extend(sent_id)
        quora_accuracy = accuracy_score(y_pred_quora, y_true_quora)
    y_true_sts = []
    y_pred_sts = []
    sents_sts1 = []
    sents_sts2 = []
    sent_ids_sts = []
    with torch.no_grad():
        for batch in tqdm(sts_dl, desc = f'STS Evaluation'):
            input_ids, attention_mask, labels, sent, sent_id = batch
            input_ids1, input_ids2 = input_ids
            attention_mask1, attention_mask2 = attention_mask
            sent1, sent2 = sent
            logits = model.predict_similarity(input_ids1, attention_mask1, input_ids2, attention_mask2)
            y_pred = logits.cpu().numpy().flatten()
            y_true = labels.cpu().numpy().flatten()
            y_true_sts.extend(y_true)
            y_pred_sts.extend(y_pred)
            sents_sts1.extend(sent1)
            sents_sts2.extend(sent2)
            sent_ids_sts.extend(sent_id)
        sts_pearson = np.corrcoef(y_pred_sts, y_true_sts)[0,1]
    print(f'SST accuracy: {sst_accuracy: .4f} Quora Accuracy: {quora_accuracy: .4f} STS Pearson Similarity: {sts_pearson: .4f}')
    return Dict(sst_accuracy= sst_accuracy, sst_pred = y_pred_sst, sst_sents = sents_sst, sst_sent_ids = sent_ids_sst, quora_accuracy = quora_accuracy,
                quora_sent1 = sents_quora1, quora_pred = y_pred_sst, quora_sent2 = sents_quora2, quora_sent_id = sent_ids_quora,
                sts_pearson = sts_pearson, sts_pred = y_pred_sts, sts_sent1 = sents_sts1, sts_sent2 = sents_sts2, sts_sent_id = sent_ids_sts,)


def MultiEvaluationTest(model, sst_dl, quora_dl, sts_dl):
    model.eval()
    y_pred_sst = []
    sents_sst = []
    sent_ids_sst = []
    with torch.no_grad():
        for batch in tqdm(sst_dl, desc = f'SST Evaluation'):
            input_ids, attention_mask, _, sent, sent_id = batch
            logits = model.predict_sentiment(input_ids, attention_mask)
            y_pred = torch.argmax(logits, dim = 1).cpu().numpy().flatten()
            y_pred_sst.extend(y_pred)
            sents_sst.extend(sent)
            sent_ids_sst.extend(sent_id)
    y_pred_quora = []
    sents_quora1 = []
    sent_ids_quora = []
    sents_quora2 = []
    with torch.no_grad():
        for batch in tqdm(quora_dl, desc = f'Quora Evaluation'):
            input_ids, attention_mask, _, sent, sent_id = batch
            input_ids1, input_ids2 = input_ids
            attention_mask1, attention_mask2 = attention_mask
            sent1, sent2 = sent
            logits = model.predict_paraphrase(input_ids1, attention_mask1, input_ids2, attention_mask2)
            y_pred = logits.cpu().numpy().flatten()
            y_pred_quora.extend(y_pred)
            sents_quora1.extend(sent1)
            sent_ids_quora.extend(sent_id)
            sents_quora2.extend(sent2)
    y_pred_sts = []
    sents_sts1 = []
    sents_sts2 = []
    sent_ids_sts = []
    with torch.no_grad():
        for batch in tqdm(sts_dl, desc = f'STS Evaluation'):
            input_ids, attention_mask, _, sent, sent_id = batch
            input_ids1, input_ids2 = input_ids
            attention_mask1, attention_mask2 = attention_mask
            sent1, sent2 = sent
            logits = model.predict_similarity(input_ids1, attention_mask1, input_ids2, attention_mask2)
            y_pred = logits.cpu().numpy().flatten()
            y_pred_sts.extend(y_pred)
            sents_sts1.extend(sent1)
            sents_sts2.extend(sent2)
            sent_ids_sts.extend(sent_id)
    return Dict(sst_pred = y_pred_sst, sst_sents = sents_sst, sst_sent_ids = sent_ids_sst, quora_pred = y_pred_quora,
                quora_sent1 = sents_quora1, quora_sent2 = sents_quora2, quora_sent_id = sent_ids_quora,
                sts_pred = y_pred_sts, sts_sent1 = sents_sts1, sts_sent2 = sents_sts2, sts_sent_id = sent_ids_sts, 
                )
