import torch.nn as nn
from transformers import AutoModel


class SBERT(nn.Module):
    """
    SentenceTransformer backbone using token-ID inputs (HuggingFace AutoModel)
    with mean pooling + custom classification head.
    Same input/output structure as your original BERT class.
    """

    def __init__(
        self,
        num_classes,
        encoder_hidden_dropout=0.1,
        encoder_attention_dropout=0.1,
        post_encoder_dropout=0.1,
        lm_weight="sentence-transformers/all-MiniLM-L6-v2",
        lm_window=512,
        **kwargs,
    ):
        super().__init__()
        self.lm_window = lm_window

        # Load backbone WITHOUT classification head
        self.lm = AutoModel.from_pretrained(
            lm_weight,
            hidden_dropout_prob=encoder_hidden_dropout,
            attention_probs_dropout_prob=encoder_attention_dropout,
        )

        hidden = self.lm.config.hidden_size

        self.dropout = nn.Dropout(post_encoder_dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def mean_pooling(self, model_output, attention_mask):
        last_hidden = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
        return pooled

    def forward(self, input):
        input_ids = input["text"]  # (batch, seq_len)

        # Check max length
        if input_ids.size(-1) > self.lm.config.max_position_embeddings:
            raise ValueError(
                f"Got maximum sequence length {input_ids.size(-1)}, "
                f"please set max_seq_length <= {self.lm.config.max_position_embeddings}"
            )

        attention_mask = (input_ids != self.lm.config.pad_token_id).long()

        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.mean_pooling(outputs, attention_mask)

        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits}
