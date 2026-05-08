import os
import random
from pathlib import Path
from typing import List, Tuple

import conllu

from constants import LANGUAGE_CODES


def read_and_combine_files(file_paths: List[Path]) -> List[conllu.TokenList]:
    """Reads multiple CoNLL-U files and combines them into a single list of sentences."""
    all_sentences = []
    valid_paths = [p for p in file_paths if p is not None and p.exists()]

    if not valid_paths:
        raise FileNotFoundError("No valid input files were provided.")

    for file_path in valid_paths:
        with file_path.open("r", encoding="utf-8") as f:
            all_sentences.extend(list(conllu.parse_incr(f)))

    return all_sentences


def filter_by_word_count(
    sentences: List[conllu.TokenList], min_words: int = 8
) -> List[conllu.TokenList]:
    """Filters sentences, keeping only those with strictly more than `min_words`."""
    valid_sentences = []

    for sent in sentences:
        # Count only standard integer IDs (ignore multi-word/empty node decimals)
        word_count = sum(1 for token in sent if isinstance(token["id"], int))
        if word_count > min_words:
            valid_sentences.append(sent)

    return valid_sentences


def sample_splits(
    sentences: List[conllu.TokenList],
    train_size: int = 100,
    dev_size: int = 20,
    test_size: int = 200,
    seed: int = 42,
) -> Tuple[List[conllu.TokenList], List[conllu.TokenList], List[conllu.TokenList]]:
    """Randomly samples sentences into Train, Dev, and Test splits."""
    total_required = train_size + dev_size + test_size

    if len(sentences) < total_required:
        raise ValueError(
            f"Need {total_required} sentences, but only have {len(sentences)}."
        )

    random.seed(seed)
    sampled = random.sample(sentences, total_required)

    # Slice into splits
    train_split = sampled[:train_size]
    dev_split = sampled[train_size : train_size + dev_size]
    test_split = sampled[train_size + dev_size :]

    return train_split, dev_split, test_split


def reindex_sentences(sentences: List[conllu.TokenList]) -> List[conllu.TokenList]:
    """Updates the sent_id metadata for a list of sentences to ensure uniqueness."""
    for i, sent in enumerate(sentences):
        sent.metadata["sent_id"] = f"{i + 1}"
    return sentences


def save_conllu_file(sentences: List[conllu.TokenList], out_path: Path):
    """Saves a list of CoNLL-U sentences to a file."""
    # Ensure the parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent.serialize())
    print(f"✅ Saved {len(sentences)} sentences to {out_path}")


def extract_text_with_conllu(conllu_path: Path, output_path: Path):
    with (
        conllu_path.open("r", encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8") as outfile,
    ):
        # parse_incr is memory efficient
        for sentence in conllu.parse_incr(infile):
            # Safely grab the text metadata
            raw_text = sentence.metadata.get("text")

            if raw_text:
                outfile.write(raw_text + "\n")
            else:
                # Fallback just in case a sentence is missing the text metadata
                print(
                    f"Warning: Sentence ID {sentence.metadata.get('sent_id')} has no '# text =' field."
                )

    print(f"Extracted sentences saved to: {output_path}")


def create_custom_dataset(
    test_path: Path, dest_dir: Path, train_path: Path = None, dev_path: Path = None
):
    """Orchestrates the pipeline to read, filter, sample, re-index, and save the dataset."""

    print("Step 1: Reading files...")
    all_sentences = read_and_combine_files([train_path, dev_path, test_path])
    print(f"-> Combined {len(all_sentences)} total sentences.")

    print("Step 2: Filtering sentences ( > 8 words )...")
    valid_sentences = filter_by_word_count(all_sentences, min_words=8)
    print(f"-> Found {len(valid_sentences)} valid sentences.")

    print("Step 3: Sampling dataset splits...")
    train_split, dev_split, test_split = sample_splits(
        sentences=valid_sentences, train_size=100, dev_size=20, test_size=200, seed=42
    )

    print("Step 4 & 5: Re-indexing and saving...")
    # Process Train
    train_split = reindex_sentences(train_split)
    train_filename = train_path.name
    save_conllu_file(train_split, dest_dir / train_filename)

    # Process Dev
    dev_split = reindex_sentences(dev_split)
    dev_filename = dev_path.name
    save_conllu_file(dev_split, dest_dir / dev_filename)

    # Process Test
    test_split = reindex_sentences(test_split)
    test_filename = test_path.name
    save_conllu_file(test_split, dest_dir / test_filename)

    print("Step 6: Creating txt files")
    train_txt_filename = train_filename.split(".")[0] + ".txt"
    extract_text_with_conllu(dest_dir / train_filename, dest_dir / train_txt_filename)

    dev_txt_filename = dev_filename.split(".")[0] + ".txt"
    extract_text_with_conllu(dest_dir / dev_filename, dest_dir / dev_txt_filename)

    test_txt_filename = test_filename.split(".")[0] + ".txt"
    extract_text_with_conllu(dest_dir / test_filename, dest_dir / test_txt_filename)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    TREEBANKS_PATH = Path("ud-treebanks-v2.17")
    SAMPLE_TREEBANK_PATH = Path("ud-treebanks-v2.17-samples")
    os.makedirs(SAMPLE_TREEBANK_PATH, exist_ok=True)

    UD_PATHS = [
        entry.name
        for entry in os.scandir(TREEBANKS_PATH)
        if entry.is_dir() and entry.name.startswith("UD_")
    ]
    exceptions = []
    for UD_PATH in UD_PATHS:
        language_code = LANGUAGE_CODES[UD_PATH.split("-")[0]]
        extra_arguments = UD_PATH.split("-")[1]
        prefix = "_".join([language_code, extra_arguments.lower()])

        train_path = TREEBANKS_PATH / f"{UD_PATH}" / f"{prefix}-ud-train.conllu"
        dev_path = TREEBANKS_PATH / f"{UD_PATH}" / f"{prefix}-ud-dev.conllu"
        test_path = TREEBANKS_PATH / f"{UD_PATH}" / f"{prefix}-ud-test.conllu"

        try:
            create_custom_dataset(
                train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                dest_dir=SAMPLE_TREEBANK_PATH / f"{UD_PATH}",
            )
        except Exception as e:
            print(e)
            exceptions.append(f"{UD_PATH}: {e}")

    print("Exceptions while runing the code: ")
    for exception in exceptions:
        print(exception)
    print(f"a total of {len(exceptions)} languages")
