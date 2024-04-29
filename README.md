# Transformer from Scratch

This project implements a basic version of the Transformer model from scratch using PyTorch. The Transformer is a popular architecture for various natural language processing tasks, including machine translation.

## Dataset
The dataset used in this project is a bilingual dataset, which contains pairs of source and target language sentences. The `BilingualDataset` class is responsible for preprocessing the dataset and preparing it for training. It utilizes tokenizers for both the source and target languages to tokenize the sentences and convert them into input sequences for the model.

## Model Inputs
The model inputs are processed and prepared within the `BilingualDataset` class. Each input sequence consists of the following components:
- `encoder_input`: The input sequence for the encoder, which includes `<s>` (start of sequence) and `</s>` (end of sequence) tokens, along with padding tokens if necessary.
- `decoder_input`: The input sequence for the decoder, which includes only the `<s>` token, along with padding tokens if necessary.
- `encoder_mask`: A mask applied to the encoder input to indicate which tokens are padding tokens.
- `decoder_mask`: A mask applied to the decoder input to prevent attending to future tokens during training.
- `label`: The target sequence for the decoder, which includes only the `</s>` token, along with padding tokens if necessary.
- `src_text` and `tgt_text`: The original source and target language sentences for reference.

## Causal Mask
The `causal_mask` function generates a mask matrix to be applied to the decoder input. This mask ensures that during training, each position can only attend to positions before it in the sequence, preventing information leakage from future tokens.

## Usage
To use this code, instantiate the `BilingualDataset` class with your dataset, source and target language tokenizers, sequence length, and language specifications. Then, you can iterate over the dataset to obtain batches of preprocessed inputs for training your Transformer model.

## Dependencies
- PyTorch
- TorchText (for tokenization)

Please note that this implementation is a simplified version of the Transformer model and may not include all features found in more complex implementations. It serves as a foundational example for understanding the core components of the Transformer architecture.
