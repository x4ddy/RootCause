# RootCause

> AI-powered debugging assistant using Retrieval-Augmented Generation (RAG)

RootCause analyzes code, retrieves similar past bug patterns, and generates **grounded bug explanations and fixes** using LLMs.

---

## Overview

RootCause is designed to improve debugging reliability by combining:
- **LLM reasoning** (for understanding code semantics)
- **Retrieval (RAG) from historical bug–fix data** (for grounding and consistency)

Instead of relying purely on model knowledge, it leverages a **continuously growing bug knowledge base**.

---

## Features

- 🔍 Context-aware bug detection  
- 🧠 Retrieval-Augmented Generation (RAG) pipeline  
- 📦 10K+ real-world bug–fix dataset  
- ⚡ FastAPI backend with caching  
- 💻 CLI + Web interface  
- 📊 Evaluated against LLM-only baseline  

---

## 🧱 System Architecture

![Architecture](./docs/architecture.png)


---

## 🔄 RAG Pipeline

![RAG Flow](./docs/rag_flow.png)

1. User submits code and/or error trace  
2. Input is normalized into structured query format  
3. Query is converted into embeddings  
4. Top-K similar bug cases are retrieved  
5. Retrieved context is injected into prompt  
6. LLM generates bug explanation and fix  
7. Structured output is returned  

---

## 📦 Dataset

![Dataset Pipeline](./docs/dataset_pipeline.png)

### Source
- GitHub PRs and commits containing bug fixes  

### Processing Steps
1. Extract diffs from PRs/commits  
2. Parse into before/after code  
3. Clean and normalize  
4. Convert into structured format  

### Example

```json
{
  "code_context": "return user.name",
  "issue": "user may be None",
  "fix": "add null check",
  "bug_type": "null_pointer"
}
```


## ⚙️ Tech Stack

| Layer          | Technology                          |
|----------------|-------------------------------------|
| Backend        | FastAPI (Python)                    |
| Database       | SQLite                              |
| Cache          | Redis                               |
| Vector Store   | FAISS / sqlite-vss                  |
| LLM            | Gemini / Groq / OpenRouter          |
| Embeddings     | HuggingFace / Google Embeddings     |
| Parsing        | AST / Regex                         |
| Interface      | CLI + Web (HTML + Tailwind)         |

## 📊 Evaluation

![Evaluation](./docs/evaluation.png)

The system is evaluated against an LLM-only baseline using a held-out test set.

| Metric              | Baseline | RAG |
|--------------------|----------|-----|
| Accuracy           |          |     |
| Fix Quality        |          |     |
| Hallucination Rate |          |     |
| Consistency        |          |     |

### Evaluation Setup

- Dataset: Held-out bug samples (not used in retrieval)
- Comparison: LLM-only vs RAG pipeline
- Method: LLM-as-judge + structured scoring
