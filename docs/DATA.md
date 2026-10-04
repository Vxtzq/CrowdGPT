# 📚 Contribute Training Data

CrowdGPT is trained on a decentralized, permissionless compute swarm. But the **data** must be high-quality and curated.

Our global dataset lives on HuggingFace: **[Vxtzq/CrowdGPT](https://huggingface.co/datasets/Vxtzq/CrowdGPT)**

## 🛡️ Is `.bin` safe?
**Yes, from a security standpoint.** A `.bin` file is just raw bytes (specifically, pre-tokenized `uint16` integers). It cannot execute malicious code like a `.py` or `.exe` file. 
**However**, the *content* of the tokens matters. We review all submissions to ensure the underlying text is high-quality, legal, and free of severe toxicity or PII. We do not train on unmoderated noise.

## 📦 The Format
To ensure the swarm trains efficiently without downloading massive text files or running tokenizers on the fly, we use **10MB pre-tokenized binary shards**.
- **Format:** Raw binary (`.bin`)
- **Dtype:** `uint16` (supports GPT-2 vocab size of 50,257)
- **Shard Size:** Exactly 1 MB (`1,048,576` bytes) per file.
- **Sequence Length:** Tokens are packed continuously. The client automatically handles the sliding window (2048 tokens per sample).

---

## 🛠️ How to contribute data

Just make a pull request on https://huggingface.co/datasets/Vxtzq/CrowdGPT-Pending.

Your data will be tokenized, and merged to the global dataset eventually.


