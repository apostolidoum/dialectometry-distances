# script to romanize Arabic using the Translit field instead of uroman
import os

from conllu import parse_incr


def transliterate_text(input_path, output_path):
    updated_sentences = []

    with open(input_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            sentence_tokens = []

            for token in tokenlist:
                # Skip multi-word token ranges (e.g., "11-13")
                if isinstance(token["id"], tuple):
                    continue

                misc = token.get("misc") or {}
                translit = misc.get("Translit")
                upos = token.get("upos")

                # Update the token form (text field)
                if translit and translit != "_":
                    token["form"] = translit
                    sentence_tokens.append(translit)
                elif translit == "_" and upos == "PUNCT":
                    token["form"] = "_"
                    sentence_tokens.append("_")
                else:
                    # Fallback to the original form if no transliteration rule applies
                    sentence_tokens.append(token["form"])

            # Update the sentence-level # text metadata comment
            if "text" in tokenlist.metadata:
                tokenlist.metadata["text"] = " ".join(sentence_tokens)

            updated_sentences.append(tokenlist)

    # Serialize back into CoNLL-U format (which outputs the updated # text line automatically)
    output_text = "".join([tokenlist.serialize() for tokenlist in updated_sentences])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)


def extract_text_comments_to_txt(conllu_path, output_txt_path):
    sentences = []

    with open(conllu_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            # Check if the 'text' key exists in the metadata comments
            if "text" in tokenlist.metadata:
                sentences.append(tokenlist.metadata["text"])

    # Write the extracted sentences to a plain text file
    with open(output_txt_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(sentences))


# Usage:
if __name__ == "__main__":
    original_folder = "30_langs/UD_Arabic-PADT"
    dest_folder = "UD_Arabic-PADTROM"

    os.makedirs(dest_folder, exist_ok=True)

    train_conllu = "ar_padt-ud-train.conllu"
    train_txt = "ar_padt-ud-train.txt"

    transliterate_text(
        os.path.join(original_folder, train_conllu),
        os.path.join(dest_folder, train_conllu),
    )
    extract_text_comments_to_txt(
        os.path.join(dest_folder, train_conllu),
        os.path.join(dest_folder, train_txt),
    )

    dev_conllu = "ar_padt-ud-dev.conllu"
    dev_txt = "ar_padt-ud-dev.txt"

    transliterate_text(
        os.path.join(original_folder, dev_conllu),
        os.path.join(dest_folder, dev_conllu),
    )
    extract_text_comments_to_txt(
        os.path.join(dest_folder, dev_conllu),
        os.path.join(dest_folder, dev_txt),
    )

    test_conllu = "ar_padt-ud-test.conllu"
    test_txt = "ar_padt-ud-test.txt"

    transliterate_text(
        os.path.join(original_folder, test_conllu),
        os.path.join(dest_folder, test_conllu),
    )
    extract_text_comments_to_txt(
        os.path.join(dest_folder, test_conllu),
        os.path.join(dest_folder, test_txt),
    )
