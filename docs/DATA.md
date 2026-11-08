# 📚 Contribute Training Data

CrowdGPT is trained on a decentralized, permissionless compute swarm. But the **data** must be high-quality and curated.

Our global dataset lives on HuggingFace: **[Vxtzq/CrowdGPT](https://huggingface.co/datasets/Vxtzq/CrowdGPT)**

## 🛡️ Is `.bin` safe?
**Yes, from a security standpoint.** A `.bin` file is just raw bytes (specifically, pre-tokenized `uint32` integers). It cannot execute malicious code like a `.py` or `.exe` file. 
**However**, the *content* of the tokens matters. We review all submissions to ensure the underlying text is high-quality, legal, and free of severe toxicity or PII. We do not train on unmoderated noise.

## 📦 The Format
To ensure the swarm trains efficiently without downloading massive text files or running tokenizers on the fly, we use **pre-tokenized binary shards**.
- **Format:** Raw binary (`.bin`), stored as `chunks/chunk_XXXX.bin` in the dataset repo
- **Tokenizer:** Qwen3 (`Qwen/Qwen3-0.6B`, vocab size 151,669)
- **Dtype:** `uint32` (the Qwen3 vocab does not fit in `uint16`)
- **Shard Size:** Several hundred MB per file. Clients don't download whole shards: they fetch 10 MB slices with HTTP range requests.
- **Sequence Length:** Tokens are packed continuously. The client automatically handles the sliding window (2048 tokens per sample).

---

## 🛠️ How to contribute data

Just make a pull request on https://huggingface.co/datasets/Vxtzq/CrowdGPT-Pending with plain text (`.txt`) files.

Or submit data at https://www.crowdgpt.net/#submit

You don't need to tokenize anything yourself: your data will be tokenized, and merged to the global dataset eventually.


