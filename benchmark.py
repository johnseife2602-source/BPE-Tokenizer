"""
Benchmark Suite: The "Token Tax" on Ge'ez / Tigrigna Script.

Compares our Custom Ge'ez-Optimized BPE Tokenizer (vocab_size=2,000)
against standard OpenAI GPT-4 (tiktoken cl100k_base, vocab_size=100,000).

Metrics Measured:
1. Token Count: Total sequence length generated.
2. Compression Ratio: UTF-8 Bytes / Token Count (Higher is better).
3. Fertility Rate: Tokens / Word (Lower is better, closer to ~1.0).
4. The Token Tax Factor: GPT-4 Tokens / Our Tokens (Shows how many times more expensive GPT-4 is).
5. Context Window & API Cost Savings: Percentage reduction in token usage.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from tokenizer import Tokenizer

# Try importing tiktoken and loading GPT-4 encoding for live benchmark comparison
try:
    import tiktoken
    try:
        gpt4_encoder = tiktoken.get_encoding("cl100k_base")
        TIKTOKEN_AVAILABLE = True
    except Exception:
        # tiktoken downloads cl100k_base.tiktoken on its first run; needs internet once to cache
        TIKTOKEN_AVAILABLE = False
        gpt4_encoder = None
except ImportError:
    TIKTOKEN_AVAILABLE = False
    gpt4_encoder = None


def word_count(text: str) -> int:
    """Calculates word count taking into account both ASCII spaces and native Ge'ez word divider (፡)."""
    # Replace native Ethiopic word separator with space for accurate word counting
    normalized = text.replace("፡", " ")
    return len([w for w in normalized.split() if w.strip()])


def run_benchmark():
    print("=" * 80)
    print("      THE TOKEN TAX BENCHMARK: GE'EZ SCRIPT OPTIMIZATION")
    print("=" * 80)

    # 1. Load our trained model
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "data" / "tigrigna_bpe.model"
    
    if not model_path.exists():
        print(f"Error: Trained model not found at {model_path}!")
        print("Please run `python main.py` first to train and save the model.")
        return

    tokenizer = Tokenizer()
    tokenizer.load(model_path)
    our_vocab_size = len(tokenizer.vocabs)
    
    print(f"Model A (Ours)  : Custom Ge'ez BPE (Vocab: {our_vocab_size:,})")
    if TIKTOKEN_AVAILABLE:
        print("Model B (GPT-4) : OpenAI cl100k_base (Vocab: 100,277) via tiktoken")
    else:
        print("Model B (GPT-4) : [tiktoken not installed - run `pip install tiktoken` for live comparison]")
    print("=" * 80)

    # 2. Benchmark Test Suite across Diverse Linguistic Domains
    test_suite = [
        {
            "domain": "Everyday Conversation",
            "text": "ሰላም ከመይ ኣለኻ? ደሓን ዲኻ? ኩሉ ነገር ጽቡቕ ድዩ ዘሎ?"
        },
        {
            "domain": "News / European Politics",
            "text": "ኣብ ዝተፈላለያ ሃገራት ኤውሮጳ ዝርከቡ ሰራሕተኛታት ሰላማዊ ሰልፊ ኣካይዶም።"
        },
        {
            "domain": "Literature / Proverb",
            "text": "ሓደ ቆልዓ ብዓዲ ይዓቢ፡ ንጽባሕ ዝኸውን ሰብ ድማ ብትምህርቲ ይስራሕ። እዚ ናይ ኣቦታትና ጥንታዊ ምኽሪ እዩ።"
        },
        {
            "domain": "Science & Technology",
            "text": "ቴክኖሎጂን ኮምፒተርን ኣብ ዘመናዊ ዓለም ዓቢ ለውጢ ኣምጺኦም ኣለዉ። ሰብ-ሰራሕ ብልሒ ድማ ንህይወት ደቂ ሰባት ይቕይሮ ኣሎ።"
        },
        {
            "domain": "Legal / Formal Affairs",
            "text": "ቤት ፍርዲ ነቲ ጉዳይ ብዕቱብ መርሚሩ ፍትሓዊ ውሳነ ክህብ ምዃኑ ብዕሊ ኣፍሊጡ ኣሎ።"
        },
        {
            "domain": "Continuous Discourse (Long)",
            "text": "እቲ ናይ መወዳእታን ዝበለጸን ናይ ግዳም ክብሪ ናይ ኣምላኽ ኣብ ናይ ፍጥረታቱ ምሉእ ሓጐስ እዩ እሞ እቲ ናይ ግዳማዊ ክብሪ ምሉእ ብምሉእ ናይ ኣምላኽ ኢዩ።"
        },
        {
            "domain": "Cross-Lingual Reference (English)",
            "text": "Workers across various European countries held peaceful demonstrations today."
        }
    ]

    total_bytes = 0
    total_words = 0
    total_our_tokens = 0
    total_gpt4_tokens = 0

    print(f"\n{'Domain':<26} | {'Bytes':<6} | {'Words':<6} | {'Ours':<6} | {'GPT-4':<6} | {'Compression':<11} | {'Token Tax':<9}")
    print("-" * 80)

    for case in test_suite:
        domain = case["domain"]
        text = case["text"]
        
        n_bytes = len(text.encode("utf-8"))
        n_words = word_count(text)
        
        # Our Tokenizer
        our_tokens = tokenizer.encode(text)
        n_our = len(our_tokens)
        our_comp = n_bytes / n_our
        our_fertility = n_our / max(1, n_words)

        total_bytes += n_bytes
        total_words += n_words
        total_our_tokens += n_our

        # GPT-4 Tokenizer (if available)
        if TIKTOKEN_AVAILABLE:
            gpt4_tokens = gpt4_encoder.encode(text)
            n_gpt4 = len(gpt4_tokens)
            total_gpt4_tokens += n_gpt4
            tax_factor = n_gpt4 / n_our
            tax_str = f"{tax_factor:.2f}x"
            gpt4_str = str(n_gpt4)
        else:
            gpt4_str = "N/A"
            tax_str = "N/A"

        # Truncate domain name for table alignment
        short_domain = (domain[:24] + "..") if len(domain) > 26 else domain
        print(f"{short_domain:<26} | {n_bytes:<6} | {n_words:<6} | {n_our:<6} | {gpt4_str:<6} | {our_comp:>5.2f}x bytes | {tax_str:>8}")

    print("-" * 80)

    # 3. Aggregate Summary & Findings
    print("\n" + "=" * 80)
    print("      AGGREGATE SUMMARY & TOKEN TAX ANALYSIS")
    print("=" * 80)
    
    overall_our_comp = total_bytes / total_our_tokens
    overall_our_fertility = total_our_tokens / total_words

    print(f"Total Test Set Volume    : {total_bytes:,} UTF-8 bytes | {total_words:,} words")
    print(f"Our Tokenizer Output     : {total_our_tokens:,} tokens")
    print(f"  • Compression Ratio    : {overall_our_comp:.2f} bytes / token")
    print(f"  • Fertility Rate       : {overall_our_fertility:.2f} tokens / word")

    if TIKTOKEN_AVAILABLE and total_gpt4_tokens > 0:
        overall_gpt4_comp = total_bytes / total_gpt4_tokens
        overall_gpt4_fertility = total_gpt4_tokens / total_words
        overall_tax = total_gpt4_tokens / total_our_tokens
        cost_savings = (1.0 - (total_our_tokens / total_gpt4_tokens)) * 100

        print(f"\nGPT-4 Output             : {total_gpt4_tokens:,} tokens")
        print(f"  • Compression Ratio    : {overall_gpt4_comp:.2f} bytes / token")
        print(f"  • Fertility Rate       : {overall_gpt4_fertility:.2f} tokens / word")
        
        print("\n" + "*" * 80)
        print(f"  🚀 THE TOKEN TAX FACTOR  : {overall_tax:.2f}x")
        print(f"  💰 CONTEXT & COST SAVINGS: {cost_savings:.1f}%")
        print("*" * 80)
        print(f"Conclusion: For the exact same information, standard GPT-4 forces Tigrigna speakers")
        print(f"to pay {overall_tax:.2f}x more tokens and shortens their effective context window by {cost_savings:.1f}%.")
        print(f"Our Ge'ez-optimized BPE eliminates this penalty completely with just {our_vocab_size:,} vocabulary entries!")
    else:
        print("\n[Tip] Run `pip install tiktoken` in your terminal to see the live GPT-4 comparison metrics!")


if __name__ == "__main__":
    run_benchmark()
