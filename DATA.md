# 📚 Contribute Training Data

CrowdGPT is trained on a decentralized, permissionless compute swarm. But the **data** must be high-quality and curated.

Our global dataset lives on HuggingFace: **[Vxtzq/CrowdGPT](https://huggingface.co/datasets/Vxtzq/CrowdGPT)**

## 🛡️ Is `.bin` safe?
**Yes, from a security standpoint.** A `.bin` file is just raw bytes (specifically, pre-tokenized `uint16` integers). It cannot execute malicious code like a `.py` or `.exe` file. 
**However**, the *content* of the tokens matters. We review all submissions to ensure the underlying text is high-quality, legal, and free of severe toxicity or PII. We do not train on unmoderated noise.

## 📦 The Format
To ensure the swarm trains efficiently without downloading massive text files or running tokenizers on the fly, we use **1MB pre-tokenized binary shards**.
- **Format:** Raw binary (`.bin`)
- **Dtype:** `uint16` (supports GPT-2 vocab size of 50,257)
- **Shard Size:** Exactly 1 MB (`1,048,576` bytes) per file.
- **Sequence Length:** Tokens are packed continuously. The client automatically handles the sliding window (64 tokens per sample).

---

## 🛠️ Step 1: Tokenize Your Data (For Contributors)

You don't need to manually shard your data. Just convert your raw text (`.txt`, `.json`, `.md`) into a single, massive `.bin` file. 

Run this Python script locally to tokenize your dataset:

~~~python
# save as tokenize.py
from transformers import GPT2TokenizerFast
import numpy as np
import os
import glob

# 1. Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 2. Find all your text files
text_files = glob.glob("my_raw_data/**/*.txt", recursive=True)
all_tokens = []

print(f"Found {len(text_files)} files. Tokenizing...")
for i, file_path in enumerate(text_files):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Tokenize and add the special end-of-text token (50256)
    tokens = tokenizer.encode(text) + [50256]
    all_tokens.extend(tokens)
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(text_files)} files...")

# 3. Convert to uint16 and save as a single massive .bin file
print("Saving to dataset.bin...")
token_array = np.array(all_tokens, dtype=np.uint16)
token_array.tofile("dataset.bin")

file_size_mb = os.path.getsize("dataset.bin") / (1024 * 1024)
print(f"✅ Done! Created dataset.bin ({file_size_mb:.2f} MB, {len(all_tokens):,} tokens)")
~~~

---

## 📤 Step 2: Submit Your `.bin`

1. Upload your `dataset.bin` to a file host, or open a Pull Request / Discussion on the **[Vxtzq/CrowdGPT HuggingFace Repository](https://huggingface.co/datasets/Vxtzq/CrowdGPT/discussions)**.
2. Include a brief description of the data source (e.g., "Public domain sci-fi novels", "My personal coding tutorials", "Wikipedia dumps").
3. The maintainers will review the text source for quality and safety.
