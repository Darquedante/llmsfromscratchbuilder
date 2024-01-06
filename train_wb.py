from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb
import torchmetrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Greedily decodes a sequence from the model's output.

    Args:
    model: The transformer model used for translation.
    source (torch.Tensor): The encoded source sequence.
    source_mask (torch.Tensor): The mask for the source sequence.
    tokenizer_src (Tokenizer): The tokenizer for the source language.
    tokenizer_tgt (Tokenizer): The tokenizer for the target language.
    max_len (int): The maximum length of the decoded sequence.
    device (torch.device): The device to run the decoding on.

    Returns:
    torch.Tensor: The decoded sequence.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output for efficiency
    encoder_output = model.encode(source, source_mask)
    # Start the decoding process with the [SOS] token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            # Stop decoding if max length is reached
            break

        # Build the target mask for the current length
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the model for the current step
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Predict the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            # Stop decoding if [EOS] token is predicted
            break

    return decoder_input.squeeze(0)
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    """
    Runs validation on the provided dataset and logs metrics using wandb.

    Args:
    model: The transformer model used for validation.
    validation_ds (DataLoader): The validation dataset.
    tokenizer_src (Tokenizer): The tokenizer for the source language.
    tokenizer_tgt (Tokenizer): The tokenizer for the target language.
    max_len (int): The maximum length for the decoded sequences.
    device (torch.device): The device to run the validation on.
    print_msg (function): Function to print messages.
    global_step (int): The current global step for logging purposes.
    num_examples (int, optional): Number of examples to show. Defaults to 2.
    """
    model.eval()
    count = 0

    # Lists to store source texts, expected translations, and model predictions
    source_texts, expected, predicted = [], [], []

    try:
        # Attempt to get the console window width for formatting
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # Default to 80 if unable to determine console width
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            # Prepare inputs and masks
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # Ensure batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Decode the model's output
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Extract texts from batch and decode prediction
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Append to lists
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print example translations
            print_msg('-'*console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    # Calculate and log various metrics
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

def get_all_sentences(ds, lang):
    """
    Generator to iterate over all sentences in the dataset for a specific language.

    Args:
    ds: The dataset to iterate over.
    lang (str): The language for which sentences are to be extracted.

    Yields:
    str: A sentence in the specified language.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves or builds a tokenizer for the specified language.

    Args:
    config (dict): Configuration dictionary containing paths and settings.
    ds: The dataset to use for training the tokenizer.
    lang (str): The language for which the tokenizer is to be built.

    Returns:
    Tokenizer: The tokenizer for the specified language.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Build a new tokenizer if not already available
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load tokenizer from file
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    Retrieves and prepares the training and validation datasets.

    Args:
    config (dict): Configuration dictionary containing dataset settings.

    Returns:
    Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]: The training dataloader, validation dataloader,
                                                          source language tokenizer, and target language tokenizer.
    """
    # Load and split the dataset
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split dataset into training and validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create BilingualDataset objects
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src, max_len_tgt = 0, 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Create dataloaders for training and validation datasets
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Creates and returns a transformer model based on the provided configuration and vocabulary sizes.

    Args:
    config (dict): Configuration dictionary containing model parameters.
    vocab_src_len (int): The size of the source language vocabulary.
    vocab_tgt_len (int): The size of the target language vocabulary.

    Returns:
    nn.Module: The constructed transformer model.
    """
    # Construct and return the transformer model
    return build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])

def train_model(config):
    """
    Trains a transformer model based on the provided configuration.

    Args:
    config (dict): Configuration dictionary containing training parameters.
    """
    # Set up the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Ensure the model weights directory exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Load datasets and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Load pre-existing model weights if specified
    initial_epoch, global_step = 0, 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Configure wandb for logging
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # Prepare inputs and masks
            encoder_input, decoder_input = batch['encoder_input'].to(device), batch['decoder_input'].to(device)
            encoder_mask, decoder_mask = batch['encoder_mask'].to(device), batch['decoder_mask'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Compute loss and update model
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # Validation after each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # Save model weights
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'global_step': global_step}, model_filename)

if __name__ == '__main__':
    # Suppress warnings and initialize configuration
    warnings.filterwarnings("ignore")
    config = get_config()
    config['num_epochs'] = 30
    config['preload'] = None

    # Initialize wandb for experiment tracking
    wandb.init(project="pytorch-transformer", config=config)
    
    # Start model training
    train_model(config)
