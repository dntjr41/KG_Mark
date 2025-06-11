# 🧬 KG-Watermark: Knowledge Graph-based Semantic Watermarking for LLMs

> This repository contains the official implementation of our **Knowledge Graph-based semantic watermarking method for LLMs**, which preserves meaning, ensures robustness, and enables provenance tracking by embedding structured knowledge into LLM outputs.

---

## 🚀 Overview

🔒 **Why watermark?**  
With the rise of open-access LLMs, detecting and attributing generated text has become a crucial problem.  
However, existing watermarking methods often:

- Rely on token-level logit manipulation (hard to access in API-based LLMs)
- Damage fluency or semantic meaning
- Are fragile under paraphrasing or translation attacks

🧠 **Our solution: KG-based Watermarking**  
We embed **semantically relevant triples from a Knowledge Graph (e.g., Wikidata5m)** into LLM-generated text using **post-processing**. This enables:

- ✔️ Meaning-preserving watermarking
- ✔️ Robustness against text edits
- ✔️ Black-box support (no logit access needed)
- ✔️ Provenance tracking of inserted knowledge

---

## 📁 Project Structure
- Update Soon

## 🔧 Installation

```bash
git clone https://github.com/your-org/kg-watermark.git
cd kg-watermark
conda create -n kgwm python=3.10
conda activate kgwm
pip install -r requirements.txt

- spacy for POS tagging
- transformers for LLM API (e.g., GPT-3.5)
- must link to Huggingface for LLM (e.g., Llama-3.1-inst)

Metrics:
- Perplexity (GPT-3.5 LM Score)
- Log Diversity
- Relevance, Coherence, Interestingness (via GPT-3.5 voting)
- Detection Accuracy: F1, Recall, TPR@1%FPR
- Provenance Recovery Rate: % of triples correctly detected
