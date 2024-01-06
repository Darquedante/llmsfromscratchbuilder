# translate.py
from pathlib import Path
import torch
import sys
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset

def translate(sentence: str) -> str:
    """
    Translates a given sentence from source to target language using a transformer model.

    Args:
    sentence (str): The source sentence to be translated. Can be an integer (or a string representing an integer)
                    indicating an index to a sentence in the test dataset.

    Returns:
    str: The translated sentence.
    """
    # Setup device and load configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()

    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    # Build and load the transformer model
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # Check if sentence is an index and load the corresponding sentence from the dataset
    label = ""
    if sentence.isdigit():
        id = int(sentence)
        try:
            ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
            ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
            sentence = ds[id]['src_text']
            label = ds[id]["tgt_text"]
        except IndexError:
            print(f"Index {id} out of range for the dataset.")
            return ""

    # Prepare the sentence for translation
    model.eval()
    with torch.no_grad():
        source, source_mask = prepare_sentence(sentence, tokenizer_src, config['seq_len'], device)
        encoder_output = model.encode(source, source_mask)

        # Initialize translation process
        predicted_translation = init_translation(model, encoder_output, source_mask, tokenizer_tgt, device, config['seq_len'])

        # Print results
        if label: 
            print(f"ID: {id}\nSOURCE: {sentence}\nTARGET: {label}\nPREDICTED: {predicted_translation}")
        else:
            print(f"SOURCE: {sentence}\nPREDICTED: {predicted_translation}")

        return predicted_translation

def prepare_sentence(sentence: str, tokenizer: Tokenizer, seq_len: int, device: torch.device) -> (torch.Tensor, torch.Tensor):
    """
    Prepares a sentence for the translation process, including tokenization and padding.

    Args:
    sentence (str): The sentence to be prepared.
    tokenizer (Tokenizer): The tokenizer to use for tokenization.
    seq_len (int): The fixed sequence length for the model.
    device (torch.device): The device to use for tensors.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: The tokenized and padded sentence, and its mask.
    """
    # Tokenization and padding
    source = tokenizer.encode(sentence)
    source = torch.cat([
        torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64), 
        torch.tensor(source.ids, dtype=torch.int64),
        torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64),
        torch.tensor([tokenizer.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
    ], dim=0).to(device)

    # Create mask for padding
    source_mask = (source != tokenizer.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

    return source, source_mask

def init_translation(model, encoder_output, source_mask, tokenizer_tgt, device, seq_len) -> str:
    """
    Initializes and performs the translation process.

    Args:
    model: The transformer model used for translation.
    encoder_output (torch.Tensor): The output from the encoder.
    source_mask (torch.Tensor): The mask for the source sentence.
    tokenizer_tgt (Tokenizer): The target language tokenizer.
    device (torch.device): The device to use for tensors.
    seq_len (int): The fixed sequence length for the model.

    Returns:
    str: The translated sentence.
    """
    decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(encoder_output).to(device)
    predicted_translation = []

    # Generate translation word by word
    while decoder_input.size(1) < seq_len:
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_output).fill_(next_word.item()).to(device)], dim=1)

        # Append translated word
        predicted_translation.append(tokenizer_tgt.decode([next_word.item()]))

        # Break on EOS token
        if next_word == tokenizer_tgt.token_to_id('[EOS]'):
            break

    return ' '.join(predicted_translation)

# Main execution
if __name__ == "__main__":
    translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good student.")

