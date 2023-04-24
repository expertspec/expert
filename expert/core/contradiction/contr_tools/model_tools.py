from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging,
)

from expert.core.contradiction.contr_tools import NLIModel
from expert.core.functional_tools import get_model_weights


logging.set_verbosity_error()


def create_model(lang: str = "en", device: str = "cpu"):
    """Function for creating the model.
    Defines model structure and download weights.

    Args:
        lang (str, optional): Speech language for text processing ['ru', 'en']. Defaults to 'en'.
        device (str, optional): Device type on local machine (GPU recommended). Defaults to 'cpu'.

    Raises:
        NotImplementedError: If 'language' is not equal to 'en' or 'ru'.

    Returns:
        [torch.model]: Model.
    """
    if lang == "en":
        model = AutoModel.from_pretrained("prajjwal1/bert-medium")

        url = "https://drive.google.com/open?id=1sJXQqnXnnJsOEbT3pbDS9c97z4x10SXm&authuser=0"
        model_name = "bert-nli-medium.pt"
        cached_file = get_model_weights(model_name=model_name, url=url)

        model = NLIModel.BERTNLIModel(model).to(device)
        model.load_state_dict(torch.load(cached_file, map_location=device))

    elif lang == "ru":
        model_checkpoint = "cointegrated/rubert-base-cased-nli-threeway"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint
        )
        model = model.to(device)

    else:
        raise NotImplementedError("'lang' must be 'en' or 'ru'.")

    return model


def choose_toketizer(lang):
    if lang == "en":
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium")
    elif lang == "ru":
        tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-base-cased-nli-threeway"
        )
    else:
        raise NameError
    return tokenizer


def get_sent1_token_type(sent):
    try:
        return [0] * len(sent)
    except ValueError:
        return []


def get_sent2_token_type(sent):
    try:
        return [1] * len(sent)
    except ValueError:
        return []


def tokenize_bert(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    return tokens


def averaging(prem_type, prem_t, hypo_t, model, tokenizer, device="cpu"):
    """Function for averaging predictions for long texts (longer than 512 tokens)."""
    func = nn.Softmax(dim=1)
    hypo_size = 512 - len(prem_t)
    parts = []
    predictions = []

    for i in range(0, len(hypo_t), hypo_size):
        parts.append(hypo_t[i : i + hypo_size])

    for part in parts:
        hypo_t = part
        hypo_type = get_sent2_token_type(hypo_t)

        indexes = prem_t + hypo_t
        indexes = tokenizer.convert_tokens_to_ids(indexes)
        indexes_type = prem_type + hypo_type
        attn_mask = get_sent2_token_type(indexes)

        indexes = torch.LongTensor(indexes).unsqueeze(0).to(device)
        indexes_type = torch.LongTensor(indexes_type).unsqueeze(0).to(device)
        attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(device)

        preds = func(model(indexes, attn_mask, indexes_type))
        predictions.append(
            [float(preds[0][0]), float(preds[0][1]), float(preds[0][2])]
        )
    # Averaging.
    predictions = torch.tensor(predictions) / len(predictions)
    prediction = torch.tensor(
        [
            predictions[:, 0].sum(),
            predictions[:, 1].sum(),
            predictions[:, 2].sum(),
        ]
    )
    prediction.unsqueeze_(0)

    return prediction


def predict_inference(
    premise: str, hypothesis: str, model, lang="en", device="cpu"
):
    """Function for prediction, returns labels:
            0 - entailment;
            1 - contradiction;
            2 - neutral.
    Args:
        premise (str): Entered text.
        hypothesis (str): Text for analysis.
        model (torch.nn.model): Get model structure and weights.
        lang (str, optional): Speech language for text processing ['ru', 'en']. Defaults to 'en'.
        device (torch.device, optional): Device type on local machine (GPU recommended). Defaults to None.

    Raises:
        NotImplementedError: If 'language' is not equal to 'en' or 'ru'.

    Returns:
        [torch.LongTensor]: Label of prediction.
    """
    if lang not in ["en", "ru"]:
        raise NotImplementedError("'lang' must be 'en' or 'ru'.")
    else:
        tokenizer = choose_toketizer(lang)

    func = nn.Softmax(dim=1)

    model.eval()
    model.to(device)

    premise = "[CLS] " + premise + " [SEP]"
    hypothesis = hypothesis + " [SEP]"

    prem_t = tokenize_bert(premise, tokenizer)
    if len(prem_t) > 512:
        return f"""
        The chosen text is too large (={len(prem_t)}).
        Sum of tokens should be less or equal to 512
        """

    hypo_t = tokenize_bert(hypothesis, tokenizer)
    if len(prem_t) + len(hypo_t) <= 512:
        prem_type = get_sent1_token_type(prem_t)
        hypo_type = get_sent2_token_type(hypo_t)

        indexes = prem_t + hypo_t
        indexes = tokenizer.convert_tokens_to_ids(indexes)
        indexes_type = prem_type + hypo_type
        attn_mask = get_sent2_token_type(indexes)

        indexes = torch.LongTensor(indexes).unsqueeze(0).to(device)
        indexes_type = torch.LongTensor(indexes_type).unsqueeze(0).to(device)
        attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(device)
        prediction = model(indexes, attn_mask, indexes_type)

        # Models have different output types.
        if lang == "en":
            return hypothesis, func(prediction).argmax()
        elif lang == "ru":
            return hypothesis, torch.softmax(prediction.logits, -1).argmax()
    else:
        hypo_size = 512 - len(prem_t)
        prem_type = get_sent1_token_type(prem_t)
        parts = hypothesis.split(".")
        parts = dict(zip(range(len(parts)), parts))

        for key, value in parts.items():
            hypo_t = tokenize_bert(value, tokenizer)
            if len(hypo_t) <= hypo_size:
                hypo_type = get_sent2_token_type(hypo_t)
                indexes = prem_t + hypo_t
                indexes = tokenizer.convert_tokens_to_ids(indexes)
                indexes_type = prem_type + hypo_type
                attn_mask = get_sent2_token_type(indexes)

                indexes = torch.LongTensor(indexes).unsqueeze(0).to(device)
                indexes_type = (
                    torch.LongTensor(indexes_type).unsqueeze(0).to(device)
                )
                attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(device)

                prediction = model(indexes, attn_mask, indexes_type)
            else:
                hypo_t = tokenize_bert(value, tokenizer)
                prediction = averaging(
                    prem_type, prem_t, hypo_t, model, tokenizer
                )
        if lang == "en":
            return hypothesis, func(prediction).argmax()
        elif lang == "ru":
            return hypothesis, torch.softmax(prediction.logits, -1).argmax()
