import random
from pathlib import Path

from conllu import parse_incr


def sort_by_master_order(sent_list, shuffled_ids):
    """Sorts sentences back into the master English sequence order using parallel_id indices."""
    return [s[1] for s in sorted(sent_list, key=lambda x: shuffled_ids.index(x[0]))]


def split_pud_by_english_keys(
    pud_dir_path: str = "pud-treebanks",
    output_dir_path: str = "pud-split",
    seed: int = 42,
) -> None:
    pud_dir = Path(pud_dir_path)
    out_dir = Path(output_dir_path)

    conllu_files = list(pud_dir.rglob("*_pud*-ud-test.conllu"))
    if not conllu_files:
        print("No PUD conllu files found.")
        return

    # 1. Identify the English file to use as the master key source
    en_file = next((f for f in conllu_files if "en_pud" in f.name), None)
    if not en_file:
        print("Could not find en_pud file to extract reference parallel_ids.")
        return

    # 2. Extract and shuffle parallel_ids from English using a fixed seed
    en_parallel_ids = []
    with open(en_file, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            pid = tokenlist.metadata.get("parallel_id")
            if pid:
                en_parallel_ids.append(pid)

    if len(en_parallel_ids) != 1000:
        print(f"[WARNING] English file has {len(en_parallel_ids)} IDs instead of 1000.")

    rng = random.Random(seed)
    shuffled_ids = list(en_parallel_ids)
    rng.shuffle(shuffled_ids)

    # 3. Define your split buckets (800 train, 100 dev, 100 test)
    train_keys = set(shuffled_ids[:800])
    dev_keys = set(shuffled_ids[800:900])
    test_keys = set(shuffled_ids[900:1000])

    print(f"Split keys defined: 800 train, 100 dev, 100 test using seed {seed}.\n")

    # 4. Process every treebank file
    for file_path in conllu_files:
        train_sentences = []
        dev_sentences = []
        test_sentences = []

        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                pid = tokenlist.metadata.get("parallel_id")
                serialized_sentence = tokenlist.serialize()

                if pid in train_keys:
                    train_sentences.append((pid, serialized_sentence))
                elif pid in dev_keys:
                    dev_sentences.append((pid, serialized_sentence))
                elif pid in test_keys:
                    test_sentences.append((pid, serialized_sentence))

        # Extract the parent treebank folder name (e.g., UD_English-PUD)
        treebank_folder_name = file_path.parent.name
        target_folder = out_dir / treebank_folder_name
        target_folder.mkdir(parents=True, exist_ok=True)

        # Get the stem name without "-ud-test" (e.g., "en_pud" from "en_pud-ud-test.conllu")
        prefix = file_path.name.replace("-ud-test.conllu", "")

        # Save Train Split (800 sentences) -> e.g., en_pud-ud-train.conllu
        train_file = target_folder / f"{prefix}-ud-train.conllu"
        with open(train_file, "w", encoding="utf-8") as f:
            f.writelines(sort_by_master_order(train_sentences, shuffled_ids))

        # Save Dev Split (100 sentences) -> e.g., en_pud-ud-dev.conllu
        dev_file = target_folder / f"{prefix}-ud-dev.conllu"
        with open(dev_file, "w", encoding="utf-8") as f:
            f.writelines(sort_by_master_order(dev_sentences, shuffled_ids))

        # Save Test Split (100 sentences) -> e.g., en_pud-ud-test.conllu
        test_file = target_folder / f"{prefix}-ud-test.conllu"
        with open(test_file, "w", encoding="utf-8") as f:
            f.writelines(sort_by_master_order(test_sentences, shuffled_ids))

        print(f"[PASS] Processed and saved splits for: {treebank_folder_name}")

    print(f"\nAll treebanks successfully split and saved into '{output_dir_path}/'!")


def create_txt_files(pud_split_dir_path: str = "pud-split") -> None:
    """
    Reads all .conllu files inside the pud-split directory structure and
    creates a corresponding .txt file for each, containing the sentence texts.
    """
    pud_split_dir = Path(pud_split_dir_path)

    # Find all conllu files recursively in the split directory layout
    conllu_files = list(pud_split_dir.rglob("*.conllu"))

    if not conllu_files:
        print(f"No .conllu files found in '{pud_split_dir_path}'.")
        return

    print(f"Found {len(conllu_files)} conllu files to convert into text files.\n")

    for file_path in conllu_files:
        sentences_text = []

        # Parse each conllu file and extract the sentence text from metadata
        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                # CoNLL-U format typically stores the raw sentence in metadata['text']
                sentence_text = tokenlist.metadata.get("text")
                if sentence_text:
                    sentences_text.append(sentence_text)

        # Define the path for the new .txt file (same folder, replacing extension)
        txt_file_path = file_path.with_suffix(".txt")

        # Write out each sentence on a new line
        with open(txt_file_path, "w", encoding="utf-8") as txt_f:
            for text in sentences_text:
                txt_f.write(text + "\n")

        print(f"[PASS] Created: {txt_file_path.relative_to(pud_split_dir.parent)}")

    print(f"\nSuccessfully generated text files for all splits!")


# To run the function, uncomment the line below:
# create_txt_files()

if __name__ == "__main__":
    split_pud_by_english_keys()
    create_txt_files()
