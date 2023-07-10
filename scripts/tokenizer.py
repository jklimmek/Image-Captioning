import argparse
import pandas as pd
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)


def create_tokenizer(path, vocab_size, limit_alph=30):
    """
    Create and train a tokenizer for text data.

    Args:
        path (str): Path to the CSV file containing the text data.
        vocab_size (int): Size of the vocabulary for the tokenizer.
        limit_alph (int, optional): Maximum size of the alphabet. Default is 30.

    Returns:
        Tokenizer: Trained tokenizer object.
    """
    # Read the CSV file and extract the text data.
    df = pd.read_csv(path, sep="|")
    txt = df["comment"].values.tolist()

    # Create a tokenizer object using WordPiece model.
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Set normalizer, pre-tokenizer, and decoder for the tokenizer.
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    # Define special tokens and initialize the WordPiece trainer.
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens, limit_alphabet=limit_alph)

    # Train the tokenizer on the text data.
    tokenizer.train_from_iterator(txt, trainer)

    # Set the post-processor for the tokenizer to handle special tokens.
    sos_token = tokenizer.token_to_id("[SOS]")
    eos_token = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[SOS]:0 $A:0 [EOS]:0",
        special_tokens=[("[SOS]", sos_token), ("[EOS]", eos_token)],
    )

    # Return the trained tokenizer.
    return tokenizer


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description='Create tokenizers.')
    parser.add_argument("--file", type=str, required=True, help="Create vocabulary from json file.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--name", type=str, required=True, help="Output file name.")
    parser.add_argument("--limit-alph", type=int, default=30, help="Limit alphabet size.")
    args = parser.parse_args()

    # Create tokenizer.
    tokenizer = create_tokenizer(args.file, args.vocab_size, args.limit_alph)

    # Save tokenizer.
    tokenizer.save(args.name)
