# script to replace the empty forms in a conllu file with _ and copy other files
import os
import shutil


def fix_empty_fields_robust(source_treebanks_path, target_treebanks_path):
    for root, dirs, files in os.walk(source_treebanks_path):
        for file in files:
            rel_path = os.path.relpath(root, source_treebanks_path)
            source_file_path = os.path.join(root, file)

            dest_dir = os.path.join(target_treebanks_path, rel_path)
            os.makedirs(dest_dir, exist_ok=True)
            target_file_path = os.path.join(dest_dir, file)

            if file.endswith(".conllu"):
                print(f"Processing CoNLL-U: {target_file_path}")

                with open(source_file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                fixed_lines = []
                for line in lines:
                    # Keep comments and empty structural boundaries intact
                    if line.startswith("#") or not line.strip():
                        fixed_lines.append(line)
                        continue

                    fields = line.split("\t")
                    # CoNLL-U format requires at least 10 tab-separated fields
                    if len(fields) >= 10:
                        # Only target FORM (index 1) and LEMMA (index 2)
                        for idx in [1, 2]:
                            if not fields[idx] or fields[idx].strip() == "":
                                fields[idx] = "_"

                        fixed_lines.append("\t".join(fields))
                    else:
                        fixed_lines.append(line)

                with open(target_file_path, "w", encoding="utf-8") as f:
                    f.writelines(fixed_lines)
            else:
                print(f"Copying file: {target_file_path}")
                shutil.copy2(source_file_path, target_file_path)


if __name__ == "__main__":
    SOURCE_PATH = "ud-treebanks-v2.17-samples-romanized"
    TARGET_PATH = "ud-treebanks-v2.17-samples-romanized-clean"

    fix_empty_fields_robust(SOURCE_PATH, TARGET_PATH)
    print("Post-processing complete! Cleaned files saved to:", TARGET_PATH)
