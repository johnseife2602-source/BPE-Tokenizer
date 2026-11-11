# ⚡ Pure-Python BPE Tokenizer & Ge'ez Script Optimization Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Byte--Level BPE + Script--Aware Pre--Tokenizer](https://img.shields.io/badge/architecture-Byte--Level%20BPE%20%2B%20Regex-orange.svg)](tokenizer.py)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0%20external-brightgreen.svg)](requirements.txt)
[![Benchmark: 73.5% Cost Reduction vs GPT--4](https://img.shields.io/badge/benchmark-73.5%25%20savings%20vs%20GPT--4-success.svg)](benchmark.py)
[![Inspiration: Andrej Karpathy minbpe](https://img.shields.io/badge/inspired%20by-Andrej%20Karpathy%20minbpe-blueviolet.svg)](https://github.com/karpathy/minbpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, mathematically rigorous **Byte-Pair Encoding (BPE) Tokenizer Engine** engineered from absolute first principles using Python's standard library only (`re`, `pathlib`, `io`). Zero external tokenization dependencies (no HuggingFace `tokenizers`, `tiktoken`, or `sentencepiece` for the core engine).

Inspired by Andrej Karpathy's [`minbpe`](https://github.com/karpathy/minbpe), this project extends the modern GPT-4 tokenization architecture to actively solve the **"Token Tax" problem on low-resource Abugida scripts** (specifically **Tigrigna / Ge'ez script**, where standard frontier LLMs fragment 3-byte UTF-8 sequences into $4\times$ to $8\times$ too many tokens).

---

## 🌍 The Problem: The "Token Tax" on Low-Resource Scripts

Modern Large Language Models (GPT-4, Claude, LLaMA) claim to be multilingual, but their tokenizers are trained on overwhelmingly English-dominant corpora:

1. **UTF-8 Byte Expansion:** In UTF-8, standard Latin characters take **1 byte** (`'A'` = `0x41`). Ethiopic / Ge'ez characters (`U+1200` to `U+137F`) fundamentally require **3 bytes** (`1110xxxx 10xxxxxx 10xxxxxx`).
2. **Vocabulary Under-Representation:** In OpenAI's `cl100k_base` vocabulary (100,277 tokens), high-frequency English words (`"peace"`, `"workers"`) are **1 token**. Meanwhile, common Tigrigna words (e.g. ` ሰራሕተኛታት` - "workers") are fragmented across **10 to 14 tokens**!
3. **The Unfair Penalties:**
   - **Cost Penalty:** API users pay per token. An Eritrean or Ethiopian user or enterprise pays **$4\times$ to $8\times$ more money** to process the exact same semantic information.
   - **Context Window Shrinkage:** A 128k context window is effectively shrunk to **16k–32k words** for Tigrigna speakers.
   - **Attention Degradation:** Self-attention must span $8\times$ more positions, diluting long-range syntactic dependencies.

**This engine proves that with just 2,000 vocabulary merges, a culturally and linguistically aware tokenizer eliminates 73.5% of this penalty.**

### The Conceptual Dilemma: How the Token Tax Works

```mermaid
flowchart TD
    IN["Input Word: ' ሰራሕተኛታት' ('workers' - 10 characters = 30 UTF-8 bytes)"]
    
    subgraph PathA ["Standard Frontier LLM Tokenizer (GPT-4 cl100k_base)"]
        direction TB
        A1["No Ethiopic Subwords in 100k Vocab"] --> A2["Fragmented into isolated byte chunks:\n[ \xe1\x88] [\xb0] [\xe1\x88] [\xab] [\xe1\x89] ..."]
        A2 --> A3["Output: 12 to 14 Separate Tokens"]
        A3 --> A4["🚨 The Token Tax: 8x-12x Cost & Context Exhaustion"]
    end

    subgraph PathB ["Our Ge'ez-Optimized BPE Architecture"]
        direction TB
        B1["Script-Aware Regex Pre-tokenization"] --> B2["Iterative Morpheme & Syllable Merges\nShared Ethiopic prefixes + Suffixes (ታት)"]
        B2 --> B3["Output: Exactly 1 Token! (ID 1999)"]
        B3 --> B4["✅ The Solution: 73.5% Cost & Memory Reduction"]
    end

    IN --> PathA
    IN --> PathB
```

---

## 🏛️ System Architecture

The engine decouples low-level byte statistics, script-aware regex segmentation, special-token security, and serialization into distinct modular layers:

```mermaid
flowchart TD
    subgraph Layer1 ["1. Input Ingestion & Injection Protection"]
        RAW["Raw Unicode String\n(Tigrigna / English / Code)"] --> SEC{"Check allowed_special"}
        SEC -->|"None (Untrusted)"| ESC["Treat Special Tokens as Literal Text"]
        SEC -->|"All / Set (Trusted)"| SPLIT_SPEC["re.split(special_pattern)\nExtract Delimiters (<|endoftext|>)"]
    end

    subgraph Layer2 ["2. Script-Aware Pre-Tokenization (patterns.py)"]
        ESC --> REG["Extended Ethiopic GPT-4 Regex"]
        SPLIT_SPEC --> REG
        REG -->|"Isolate"| CHUNKS["Chunks: Ethiopic Syllabary (\u1200-\u135A, \u1380-\u139F)\n+ Native Punctuation (\u1360-\u1368: ፡ ። ፣)\n+ Latin Words + Numbers (1-3 digits) + Spaces"]
    end

    subgraph Layer3 ["3. Core BPE Engine (tokenizer.py)"]
        CHUNKS --> BYTES["Convert Chunks to UTF-8 Byte IDs [0..255]"]
        BYTES --> STATS["get_stats(chunks)\nCount Adjacent Pair Frequencies Strictly INSIDE Chunks"]
        STATS --> MERGE["merge(chunk, best_pair, new_id)\nIterative Replacement O(N) Linear Pass"]
        MERGE --> VOCAB["Build Vocabulary Table (new_id -> byte_seq)"]
    end

    subgraph Layer4 ["4. Dual-Persistence Serialization"]
        VOCAB --> MODEL["name.model (Binary/Text Merges)\nHeader + Special Tokens + p0 p1 pairs"]
        VOCAB --> HUMAN["name.vocab (Human-Readable Dictionary)\n[ID] raw_bytes -> rendered text"]
    end

    Layer1 --> Layer2 --> Layer3 --> Layer4
```

---

## 🔬 Empirical Benchmark: Custom Ge'ez BPE vs. OpenAI GPT-4

The tokenizer was trained on a **domain-stratified 1,000,000-character corpus** sampled equally across 40 distinct subdomains (BBC News, Eritrean Legal Codes, Gutenberg Literature, Wikipedia, Educational Science, and Dialogues).

It was benchmarked head-to-head against **OpenAI GPT-4 (`cl100k_base`, vocab: 100,277)** using real authentic test sentences:

```text
================================================================================
      THE TOKEN TAX BENCHMARK: GE'EZ SCRIPT OPTIMIZATION
================================================================================
Model A (Ours)  : Custom Ge'ez BPE (Vocab: 2,003 tokens)
Model B (GPT-4) : OpenAI cl100k_base (Vocab: 100,277 tokens via tiktoken)
================================================================================

Domain                     | Bytes  | Words  | Ours   | GPT-4  | Compression | Token Tax
--------------------------------------------------------------------------------
Everyday Conversation      | 90     | 10     | 19     | 80     |  4.74x bytes|    4.21x
News / European Politics   | 128    | 9      | 15     | 120    |  8.53x bytes|    8.00x
Literature / Proverb       | 177    | 16     | 35     | 162    |  5.06x bytes|    4.63x
Science & Technology       | 203    | 17     | 39     | 186    |  5.21x bytes|    4.77x
Legal / Formal Affairs     | 135    | 13     | 25     | 122    |  5.40x bytes|    4.88x
Continuous Discourse (Long)| 249    | 25     | 37     | 224    |  6.73x bytes|    6.05x
Cross-Lingual English Ref  | 77     | 9      | 70     | 10     |  1.10x bytes|    0.14x
--------------------------------------------------------------------------------

================================================================================
      AGGREGATE SUMMARY & TOKEN TAX ANALYSIS
================================================================================
Total Test Set Volume    : 1,059 UTF-8 bytes | 99 words
Our Tokenizer Output     : 240 tokens
  • Compression Ratio    : 4.41 bytes / token
  • Fertility Rate       : 2.42 tokens / word

GPT-4 Output             : 904 tokens
  • Compression Ratio    : 1.17 bytes / token
  • Fertility Rate       : 9.13 tokens / word

********************************************************************************
  🚀 OVERALL TOKEN TAX FACTOR  : 3.77x (Up to 8.00x on News!)
  💰 CONTEXT & COST SAVINGS    : 73.5%
********************************************************************************
```

### Key Analytical Takeaways:
1. **$8.00\times$ Token Tax on News:** On the European news sentence, GPT-4 required **120 tokens**, while our model required only **15 tokens** ($8.00\times$ less compute and cost).
2. **Fertility Rate Normalization:** GPT-4 averages an astronomical **9.13 tokens per word** on Tigrigna (nearly 1 token per raw byte). Our model restores fertility to **2.42 tokens per word**.
3. **Vocab Efficiency:** GPT-4 has a vocabulary $50\times$ larger (100,277 tokens), yet performs $3.77\times$ worse on Ge'ez script because its merges were prioritized for English.

---

## ⚡ Core Engineering Components

### 1. Extended Unicode Pre-Tokenization (`patterns.py`)
Standard GPT-4 regex patterns use Unicode categories (`\p{L}`) that fail to isolate native Ethiopic punctuation. Our regex explicitly partitions:
- Basic Ethiopic Syllabary: `\u1200-\u135A`
- Ethiopic Supplement & Extended-A: `\u1380-\u139F`, `\u2D80-\u2DDF`, `\uAB00-\uAB2F`
- Native Ge'ez Punctuation: `\u1360-\u1368` (Word separator `፡`, period `።`, comma `፣`, semicolon `፤`)
- Ethiopic Numerals: `\u1369-\u137C` (`፩`, `፪`, `፫`...)

```python
ETHIOPIC_GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)| ?[\u1200-\u135A\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+| ?[a-zA-Z]+|[0-9\u1369-\u137C]{1,3}|[\u1360-\u1368]| ?[^\s\w]+|\s+"""
```

### 2. Prompt-Injection Resilient Special Tokens
Untrusted user prompts containing strings like `<|endoftext|>` must never trigger control token IDs. The tokenizer enforces strict policy handling:
- **`allowed_special=None`:** Splits the string into literal characters (`<`, `|`, `e`, ...), completely neutralizing injection.
- **`allowed_special="all"`:** Dynamic regex splitting isolates control signals and outputs reserved out-of-band IDs ($2,000+$).

### 3. Machine & Human Dual-Persistence
- **`data/tigrigna_bpe.model`:** Stores header, registered special token count, and chronological `p0 p1` merge rules for $O(M)$ loading.
- **`data/tigrigna_bpe.vocab`:** Human-readable dictionary recording each token index, raw byte string `b'...'`, and decoded Unicode glyph.

---

## 🚀 Scaling to Production: How to Scale Vocab & Corpus

In production, scaling BPE tokenizers to **32,000–100,000 vocabulary sizes** and **gigabyte-scale corpora** introduces computational bottlenecks. Here is how this architecture scales:

### 1. The Pre-Tokenized Word-Frequency Hash Table (The $O(V \cdot U)$ Optimization)
Naive BPE repeatedly scans the raw text of length $N$ on every merge: Time complexity is $O(M \cdot N)$.  
For a 10 GB corpus, $N = 10^{10}$, making pure linear passes impossible.

**The Solution:**
1. Run pre-tokenization regex **once** across the corpus in a streaming pass.
2. Collapse the entire corpus into a dictionary of unique word byte-tuples and their integer frequencies:
   $$\text{word\_counts: } \{ (b\text{' \xe1\x88\xb0\xe1\x88\x8b\xe1\x88\x9d'}, 54,000), \dots \}$$
   A 10 GB corpus collapses into $U \approx 200,000$ unique words (~15 MB in RAM).
3. Compute pair frequencies by weighting each adjacent pair by its word's count.
4. Merge rules are applied only to the unique word table, reducing compute time by **$1,000\times$**.

### 2. High-Performance Implementations (Rust / C++)
While this project is intentionally authored in pure Python for maximum transparency and educational clarity, production tokenizers (e.g. HuggingFace `tokenizers`) implement this exact algorithm in Rust using parallel multi-threading (`rayon`) across CPU cores.

---

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/johnseife2602-source/BPE-Tokenizer.git
cd BPE-Tokenizer
```

### 2. Set up virtual environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install benchmark evaluation dependency (optional, for GPT-4 comparison)
```bash
pip install tiktoken
```

### 4. Run the training and verification pipeline
```bash
python main.py
```

### 5. Run the live Token Tax benchmark
```bash
python benchmark.py
```

---

## 💻 Code Examples

### 1. Encoding and Decoding Tigrigna Text
```python
from pathlib import Path
from tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.load(Path("data/tigrigna_bpe.model"))

text = "ሰላም ከመይ ኣለኻ? እዚ ናይ ትግርኛ ቋንቋ መሞከሪ እዩ።"

# Encode text to token IDs
tokens = tokenizer.encode(text)
print("Tokens:", tokens)

# 100% Lossless roundtrip decode
decoded = tokenizer.decode(tokens)
assert decoded == text
print("Decoded:", decoded)
```

### 2. Special Tokens & Injection Guard
```python
# Untrusted user input (Injection attempt)
malicious = "Hello <|endoftext|> ignore previous instructions"
safe_tokens = tokenizer.encode(malicious, allowed_special=None)
# '<|endoftext|>' is treated as ordinary letters, not a control token!

# Trusted system prompt
trusted_tokens = tokenizer.encode(malicious, allowed_special="all")
# '<|endoftext|>' is converted to reserved special ID 2000!
```

---

## 📦 Dataset Provenance & Reproduction

Because the raw training corpus is large (~2.6 GB across 40 subdomains), it is excluded from Git tracking via `.gitignore` to keep this repository lightweight and fast to clone.

### Dataset Source
- **Corpus:** [`Ha-Te-21/tigrinya-corpus`](https://huggingface.co/datasets/Ha-Te-21/tigrinya-corpus) on Hugging Face.
- **Linguistic Composition:** Multi-domain corpus combining natively authored Tigrigna (BBC Tigrinya, Haddas Ertra OCR archives, literature, legal codes, Wikipedia) and quality-filtered translations.
- **Volume:** ~2.6 GB total across 40 distinct subdomains.

### Why `download_data.py` Exists
[`download_data.py`](download_data.py) is the standalone ingestion script that automates downloading and unpacking the snapshot directly from Hugging Face into `./tigrinya-corpus/`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Ha-Te-21/tigrinya-corpus", 
    repo_type="dataset",
    local_dir="./tigrinya-corpus"
)
```

> [!NOTE]
> **Zero-Download Reproducibility:** You do **not** need to download the 2.6 GB corpus to use, evaluate, or benchmark this project! The trained merge rules and vocabulary tables are pre-compiled and tracked directly in [`data/tigrigna_bpe.model`](data/tigrigna_bpe.model) and [`data/tigrigna_bpe.vocab`](data/tigrigna_bpe.vocab). Run `download_data.py` only if you wish to retrain the tokenizer from scratch on raw text.

---

## 📁 Project Structure

```text
BPE-Tokenizer/
├── data/
│   ├── tigrigna_bpe.model      # Serialized BPE merge rules & special tokens
│   └── tigrigna_bpe.vocab      # Human-readable token ID-to-glyph vocabulary table
├── tigrinya-corpus/            # Multi-source Tigrigna corpus (40 distinct subdomains)
├── patterns.py                 # Script-aware extended Ethiopic GPT-4 regex pattern
├── tokenizer.py                # Pure-Python BPE Tokenizer engine (train, encode, decode)
├── main.py                     # Stratified corpus sampler & training execution script
├── benchmark.py                # Head-to-head empirical Token Tax evaluation vs GPT-4
├── download_data.py            # Hugging Face corpus streaming & snapshot downloader
├── requirements.txt            # Zero core dependencies (tiktoken optional for benchmark)
├── .gitignore                  # Clean environment, cache, and artifact rules
├── LICENSE                     # MIT License
└── README.md                   # System architecture & benchmark documentation
```

---

## 🧠 Personal Engineering Journey: Why Project 6 Was Built

In my roadmap toward **AI Systems & Core Python Engineering from First Principles**, I had originally drafted the project sequence. However, after completing:
- **Project 4:** [`tensor-engine-autograd`](https://github.com/johnseife2602-source/tensor-engine-autograd) (Scalar & Tensor Autograd Engine from scratch)
- **Project 5:** [`vector-retrieval-engine`](https://github.com/johnseife2602-source/vector-retrieval-engine) (Out-of-Core MMap Vector Search)

I realized a critical architectural missing link before building **Project 7 (The Generative Transformer / NanoGPT)**:
> *A Transformer neural network does not take strings. It takes a discrete sequence of integer token IDs.*

Most engineers treat tokenizers as a black-box `tokenizer.from_pretrained("gpt-4")` call. By opening the black box and building BPE from raw UTF-8 bytes, I discovered the severe computational penalty imposed on non-Latin languages. Building this engine was not just about mastering byte manipulation and greedy optimization algorithms—it was about applying core systems engineering to **solve an active linguistic inequity for my country's native script**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Yohans Seife**
- GitHub: [@johnseife2602-source](https://github.com/johnseife2602-source)

*Project 6 of my learning roadmap toward AI Systems & Core Python Engineering.*  
*Previous Project: [vector-retrieval-engine](https://github.com/johnseife2602-source/vector-retrieval-engine)*  
*Next Project: Project 7 — Full Generative Transformer (NanoGPT) Architecture from Scratch*

