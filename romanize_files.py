import uroman as ur
import os

if __name__ == "__main__":
    uroman = ur.Uroman()

    TREEBANKS_PATH = "ud-treebanks-v2.17-samples"
    TARGET_TREEBANKS_PATH = "ud-treebanks-v2.17-samples-romanized"

    UD_PATHS = [
        entry.name
        for entry in os.scandir(TREEBANKS_PATH)
        if entry.is_dir() and entry.name.startswith("UD_")
    ]
    for path in UD_PATHS:
        print(f"Processing directory: {path}")

        source_dir = os.path.join(TREEBANKS_PATH, path)
        target_dir = os.path.join(TARGET_TREEBANKS_PATH, path)

        os.makedirs(target_dir, exist_ok=True)

        for entry in os.scandir(source_dir):
            if entry.is_file() and entry.name.endswith((".conllu", ".txt")):
                target_file_path = os.path.join(target_dir, entry.name)
                uroman.romanize_file(
                    input_filename=os.path.join(source_dir, entry.name),
                    output_filename=target_file_path,
                )
