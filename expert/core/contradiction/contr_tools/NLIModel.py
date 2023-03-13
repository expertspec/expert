import torch.nn as nn


class BERTNLIModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        output_dim = 3
        self.bert = model

        embedding_dim = self.bert.config.to_dict()["hidden_size"]
        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, sequence, attn_mask, token_type):

        embedded = self.bert(
            input_ids=sequence,
            attention_mask=attn_mask,
            token_type_ids=token_type
        )[1]
        output = self.out(embedded)

        return output
