from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoModel


class SDistilBERT(nn.Module):
    """
    SentenceTransformer backbone using token-ID inputs (HuggingFace AutoModel)
    with mean pooling + custom classification head.
    """

    def __init__(
        self,
        num_classes,
        post_encoder_dropout=0.1,
        lm_weight="sentence-transformers/msmarco-distilbert-base-v4",
        encoder_ckpt_path="",
        encoder_prefix_to_strip="embedding_labels.encoder.transformer.0.auto_model.",
        strict_encoder=True,
        **kwargs,
    ):
        super().__init__()

        # Load backbone WITHOUT classification head
        self.lm = AutoModel.from_pretrained(lm_weight)

        hidden = self.lm.config.hidden_size

        self.dropout = nn.Dropout(post_encoder_dropout)
        self.classifier = nn.Linear(hidden, num_classes)

        if encoder_ckpt_path:
            self._load_encoder_from_path(
                encoder_ckpt_path,
                prefix_to_strip=encoder_prefix_to_strip,
                strict=strict_encoder,
            )

    def _load_encoder_from_path(self, path, prefix_to_strip="", strict=True):
        print(f"Using pretrained encoder. Loading from {path}")

        old_state_dict = torch.load(path, map_location="cpu", weights_only=False)
        new_state_dict = OrderedDict()

        for k, v in old_state_dict.items():
            name = k.replace("embedding_labels.encoder.transformer.0.auto_model.", "")
            new_state_dict[name] = v

        missing, unexpected = self.lm.load_state_dict(new_state_dict, strict=strict)
        print("Encoder load_state_dict done.")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    def mean_pooling(self, model_output, attention_mask):
        last_hidden = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
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
