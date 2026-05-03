import os
import json
import pickle
import faiss
import numpy as np
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------
# 1. CONFIG (ENV + MODELS)
# ---------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-xx")
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "meta-llama/llama-3.1-8b-instruct"

# ---------------------------------------------------------
# 2. PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "train_corpus.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "train_corpus_metadata.pkl")

# ---------------------------------------------------------
# 3. MCP + CLIENT INIT
# ---------------------------------------------------------
mcp = FastMCP("RootCause")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

TRAIN_INDEX = None
TRAIN_METADATA = None


def load_assets():
    global TRAIN_INDEX, TRAIN_METADATA

    if TRAIN_INDEX is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
        TRAIN_INDEX = faiss.read_index(FAISS_INDEX_PATH)

    if TRAIN_METADATA is None:
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
        with open(METADATA_PATH, "rb") as f:
            TRAIN_METADATA = pickle.load(f)


# ---------------------------------------------------------
# 4. RERANKING
# ---------------------------------------------------------
def rerank_with_llm(query, retrieved):
    if not retrieved:
        return []

    prompt = f"""
Rank these bug candidates by relevance.

Query:
{query}

Candidates:
"""

    for i, r in enumerate(retrieved):
        prompt += f"\n[{i}] Type: {r['bug_type']}\nIssue: {r['issue'][:200]}\n"

    prompt += "\nReturn ONLY JSON list like [1,0,2]"

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        order = json.loads(response.choices[0].message.content)

        if isinstance(order, list):
            return [retrieved[i] for i in order if i < len(retrieved)]

    except Exception:
        pass

    return retrieved


# ---------------------------------------------------------
# 5. MCP TOOL
# ---------------------------------------------------------
@mcp.tool()
def analyze_bug(query: str) -> str:
    """
    Analyze a bug using RAG + reranking + fallback.
    Returns structured JSON.
    """

    try:
        load_assets()

        # ---- EMBEDDING ----
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query[:8000]
        )

        emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)

        scores, indices = TRAIN_INDEX.search(emb, 5)

        # ---- RETRIEVAL ----
        retrieved = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                sample = TRAIN_METADATA[idx]
                retrieved.append({
                    "score": float(scores[0][i]),
                    "bug_type": sample.get("bug_type"),
                    "issue": sample.get("issue"),
                    "fix": sample.get("fix"),
                    "patch": sample.get("patches", [""])[0][:500] if sample.get("patches") else ""
                })

        # ---- CONFIDENCE GATING ----
        top_score = retrieved[0]["score"] if retrieved else 0
        use_rag = top_score >= 0.6

        # ---- RERANK ----
        if use_rag:
            retrieved = rerank_with_llm(query, retrieved)

        # ---- CONTEXT ----
        context = ""
        for r in retrieved[:3]:
            context += f"""
--- EXAMPLE ---
Type: {r['bug_type']}
Issue: {r['issue']}
Fix: {r['fix']}
Patch: {r['patch']}
"""

        # ---- GENERATION ----
        system_prompt = """
        You are RootCause, a debugging agent used by coding assistants.

        Your job:
        1. Identify the root cause of the bug precisely
        2. Provide a concrete fix (code or clear instruction)
        3. Base your answer on the most relevant historical examples if provided

        Rules:
        - Be specific, not generic
        - Do NOT hallucinate APIs or behavior
        - If examples are weak, rely on reasoning instead

        Return ONLY JSON:
        {
        "root_cause": "...",
        "fix": "...",
        "confidence": 0-1,
        "examples_used": []
        }
        """

        user_prompt = (
            f"{context}\nBug: {query}"
            if use_rag
            else f"Bug: {query}\n(No strong matches found)"
        )

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------
# 6. RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")