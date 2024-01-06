# bilingual_dataset.py
import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Token initialization
        self.sos_token = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")

        # Pre-create a tensor of pad tokens for maximum length
        self.max_pad_tensor = torch.full((seq_len,), self.pad_token, dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate the number of padding tokens needed
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Slice the pre-created pad tensor to required length
        enc_padding = self.max_pad_tensor[:enc_num_padding_tokens]
        dec_padding = self.max_pad_tensor[:dec_num_padding_tokens]

        # Construct encoder and decoder inputs
        encoder_input = torch.cat(
            [torch.tensor([self.sos_token] + enc_input_tokens + [self.eos_token], dtype=torch.int64),
             enc_padding], dim=0)

        decoder_input = torch.cat(
            [torch.tensor([self.sos_token] + dec_input_tokens, dtype=torch.int64),
             dec_padding], dim=0)

        label = torch.cat(
            [torch.tensor(dec_input_tokens + [self.eos_token], dtype=torch.int64),
             dec_padding], dim=0)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": self._create_mask(encoder_input),
            "decoder_mask": self._create_mask(decoder_input) & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    def _create_mask(self, tensor):
        return (tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int()

def causal_mask(size):
    """
    Creates a causal mask to mask out future tokens in a sequence.
    This ensures that predictions for position i can depend only on known outputs at positions less than i.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
