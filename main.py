import sys
from pathlib import Path
import json
from tokenizer import Tokenizer

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def build_stratified_corpus(corpus_dir="./tigrinya-corpus", total_chars=1_000_000):
    """
    Builds a balanced, domain-stratified corpus by sampling equal quotas
    from every sub-dataset (news, books, legal, science, religious, etc.).
    """
    corpus_path = Path(corpus_dir)
    # Find all domain files inside subfolders (skipping the giant root file)
    domain_files = [p for p in corpus_path.rglob("*.jsonl") if p.parent != corpus_path]

    if not domain_files:
        raise ValueError(f"No domain files found in {corpus_dir}")

    chars_per_file = total_chars // len(domain_files)
    print(f"Sampling ~{chars_per_file:,} characters from each of the {len(domain_files)} domains...")

    gathered = []
    for file_path in domain_files:
        file_chars = 0
        with open(file_path, "r", encoding="utf-8") as f_s:
            for line in f_s:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                    if text:
                        gathered.append(text)
                        file_chars += len(text)
                        if file_chars >= chars_per_file:
                            break
                except json.JSONDecodeError:
                    continue

    blended_corpus = "\n".join(gathered)
    print(f"Blended corpus ready: {len(gathered):,} sentences ({len(blended_corpus):,} total characters).\n")
    return blended_corpus


def main():
    print("=" * 65)
    print("  PROJECT 6: GE'EZ / TIGRIGNA BPE TOKENIZER")
    print("=" * 65)

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "data" / "tigrigna_bpe.model"
    target_vocab_size = 2000

    tokenizer = Tokenizer()

    # If the model already exists on disk, load it; otherwise train a new one
    if model_path.exists():
        print(f"\n[1/3] Loading existing model from {model_path}...")
        tokenizer.load(model_path)
        print(f"Loaded successfully! Vocabulary size: {len(tokenizer.vocabs):,} tokens.")
    else:
        print("\n[1/3] Building stratified corpus across all 40 Tigrigna domains...")
        corpus = build_stratified_corpus(corpus_dir=base_dir / "tigrinya-corpus", total_chars=1_000_000)

        print(f"[2/3] Training Ge'ez-Optimized BPE (vocab_size={target_vocab_size})...")
        tokenizer.train(corpus, vocab_size=target_vocab_size, verbose=True)

        tokenizer.register_special_tokens({
            "<|endoftext|>": target_vocab_size,
            "<|im_start|>": target_vocab_size + 1,
            "<|im_end|>": target_vocab_size + 2,
        })

        print("\n[3/3] Saving trained model and vocabulary...")
        tokenizer.save("tigrigna_bpe")
        print(f"Model saved to {model_path}!")

    # Verification and Demonstration
    print("\n" + "=" * 65)
    print("  MODEL VERIFICATION & ENCODING DEMO")
    print("=" * 65)

    test_sentences = [
        "ሰላም ከመይ ኣለኻ? ደሓን ዲኻ?",
        "ኣነ ዮሃንስ ይብሃል። ንስኪ መን ትበሃሊ?",
        "ኣብ ዝተፈላለያ ሃገራት ኤውሮጳ ዝርከቡ ሰራሕተኛታት ሰላማዊ ሰልፊ ኣካይዶም።",
        "Hello world! This is a bilingual test <|endoftext|> ሰላም ዓለም።"
    ]

    for i, sentence in enumerate(test_sentences, 1):
        encoded = tokenizer.encode(sentence, allowed_special="all")
        decoded = tokenizer.decode(encoded)
        raw_bytes = len(sentence.encode("utf-8"))
        num_tokens = len(encoded)
        compression = raw_bytes / num_tokens

        assert decoded == sentence, f"Lossless check failed for: {sentence}"
        print(f"\n[Sample {i}] '{sentence}'")
        print(f"  Raw UTF-8 Bytes : {raw_bytes} bytes")
        print(f"  Token IDs ({num_tokens}) : {encoded[:12]}{'...' if len(encoded) > 12 else ''}")
        print(f"  Compression     : {compression:.2f}x bytes/token")
        print(f"  Decoded Match   : [PASS - 100% Lossless]")


if __name__ == "__main__":
    main()
