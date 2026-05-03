import os
import json
import time
import pickle
import logging
import numpy as np
import faiss
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

# --- Configuration & Auth ---
API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-xx")
PARSER_MODEL = "microsoft/phi-4" 
EMBED_MODEL = "openai/text-embedding-3-small"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# 1. CONSTANTS & MAPPINGS
# =========================================================
CANONICAL_BUGS = {
    "logic_bug", "api_misuse", "type_error", "exception_handling", 
    "null_check", "scope_error", "off_by_one", "concurrency", 
    "memory_leak", "race_condition", "security", "performance"
}

NORMALIZATION_MAP = {
    "logics_bug": "logic_bug", "algorithmic_bug": "logic_bug", "out_of_bounds": "off_by_one",
    "infinite_recursion": "logic_bug", "division_by_zero": "logic_bug", "api_ misuse": "api_misuse",
    "API_MISUSE": "api_misuse", "deprecated_api_misuse": "api_misuse", "timezone_misuse": "api_misuse",
    "data_type": "type_error", "encoding": "type_error", "key_error": "type_error",
    "error_handling": "exception_handling", "null": "null_check", "none": "null_check",
    "invalid_reference": "null_check", "name_error": "scope_error", "security_issue": "security",
    "memory_error": "memory_leak", "performance_bug": "performance", "caching_error": "performance",
    "serialization": "api_misuse", "attribute_error": "type_error", "bug": "logic_bug"
}

PARSER_PROMPT = """You are an expert Python Bug Analyst and Data Extraction Tool. 
Your singular task is to analyze a GitHub code diff and commit message, and output a strict JSON object mapping the bug to a predefined schema.

CRITICAL INSTRUCTIONS:
1. Output ONLY valid JSON. 
2. Do NOT wrap the JSON in markdown code blocks (no ```json). 
3. Do NOT include greetings, explanations, or any conversational text. 
4. Your response must begin exactly with `{` and end exactly with `}`.
5. You MUST map the 'bug_type' strictly to ONE of the allowed ENUM values below. Do not invent new types.

BUG TYPE ENUMERATIONS (Choose exactly one):
- "null_check": Missing or incorrect 'None' checks.
- "type_error": Type mismatches, casting errors, KeyError, AttributeError.
- "logic_bug": Incorrect algorithms, wrong math, flawed boolean logic, or bad state management.
- "off_by_one": Loop boundary errors, list index out of range.
- "scope_error": Variable scoping issues, NameError, shadowing.
- "exception_handling": Bad try/except blocks, catching wrong exceptions.
- "api_misuse": Incorrect library function usage, wrong parameters passed to an API.
- "concurrency": Race conditions, missing thread locks, async/await issues.
- "performance": Inefficient loops, N+1 queries, memory bloat.
- "security": Unsafe deserialization, SQL injection, unsafe regex.

SCHEMA DEFINITION:
{
  "bug_type": "<string> (Must be exact match from the enum list above)",
  "issue": "<string> (A clear, technical, single-sentence explanation of what was broken)",
  "fix": "<string> (A clear, technical, single-sentence explanation of how the patch fixed it)",
  "severity": "<string> (Must be 'low', 'medium', or 'high')",
  "confidence": "<string> (Must be 'low', 'medium', or 'high' based on how clearly you understand the diff)"
}"""

class RootCausePipeline:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY
        )
        self.parse_model = PARSER_MODEL
        self.embed_model = EMBED_MODEL
        self.parse_workers = 10
        self.embed_workers = 10

    def _parse_sample(self, sample: Dict) -> Dict:
        title = sample.get("title") or sample.get("message", "")
        patches = "\n".join(sample.get("patches", []))
        user_content = f"Title: {title}\n\nDiff:\n{patches[:3000]}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.parse_model,
                messages=[
                    {"role": "system", "content": PARSER_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=300
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                truncated = raw[:raw.rfind("}")+1]
                parsed = json.loads(truncated)
                
            return {**sample, **parsed}
        except Exception as e:
            return {**sample, "parse_error": str(e)}

    def _normalize_bug_type(self, bug_type: str) -> str:
        if not bug_type:
            return "logic_bug"
        bt_clean = bug_type.strip()
        if bt_clean in CANONICAL_BUGS:
            return bt_clean
        if bt_clean in NORMALIZATION_MAP:
            return NORMALIZATION_MAP[bt_clean]
        bt_lower = bt_clean.lower().replace(" ", "_").replace("-", "_")
        for canonical in CANONICAL_BUGS:
            if canonical in bt_lower or bt_lower in canonical:
                return canonical
        return "logic_bug"

    def _fetch_embedding(self, text: str) -> Optional[List[float]]:
        for attempt in range(3):
            try:
                res = self.client.embeddings.create(model=self.embed_model, input=text[:8000])
                return res.data[0].embedding
            except Exception:
                time.sleep(1)
        return None

    def run_pipeline(self, raw_input_file: str, output_dir: str):
        logger.info(f"Starting RootCause ETL Pipeline on {raw_input_file}")
        
        # 1. Load Raw
        raw_samples = []
        with open(raw_input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_samples.append(json.loads(line))
        
        # 2. Parse via LLM
        logger.info(f"Phase 1: Parsing {len(raw_samples)} samples...")
        parsed_samples = []
        with ThreadPoolExecutor(max_workers=self.parse_workers) as executor:
            futures = [executor.submit(self._parse_sample, s) for s in raw_samples]
            for future in as_completed(futures):
                res = future.result()
                if "parse_error" not in res and res.get("confidence") != "low":
                    parsed_samples.append(res)
        
        # 3. Normalize & Save Dataset to Device
        logger.info("Phase 2: Normalizing & Saving Dataset...")
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = os.path.join(output_dir, "parsed_bug_corpus.jsonl")
        
        with open(dataset_path, "w", encoding="utf-8") as f:
            for sample in parsed_samples:
                sample["bug_type"] = self._normalize_bug_type(sample.get("bug_type"))
                f.write(json.dumps(sample) + "\n")
        
        logger.info(f"Normalized dataset saved to: {dataset_path}")

        # 4. Generate Embeddings (Using the normalized data)
        logger.info(f"Phase 3: Embedding {len(parsed_samples)} samples...")
        texts = [
            f"Bug type: {s.get('bug_type')}\nIssue: {s.get('issue')}\nFix: {s.get('fix')}\nContext: {s.get('title')}\n" 
            for s in parsed_samples
        ]
        
        embeddings = []
        valid_samples = []
        with ThreadPoolExecutor(max_workers=self.embed_workers) as executor:
            futures = {executor.submit(self._fetch_embedding, t): i for i, t in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                emb = future.result()
                if emb:
                    embeddings.append(emb)
                    valid_samples.append(parsed_samples[idx])

        # 5. Build FAISS
        if not embeddings:
            logger.error("Phase 4 Failed: No embeddings generated.")
            return

        logger.info("Phase 4: Constructing FAISS Index...")
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embedding_matrix)
        
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        # 6. Save FAISS Artifacts
        faiss.write_index(index, os.path.join(output_dir, "train_corpus.faiss"))
        with open(os.path.join(output_dir, "train_corpus_metadata.pkl"), "wb") as f:
            pickle.dump(valid_samples, f)
            
        logger.info("Pipeline Complete! Vector index and Metadata saved.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_file = os.path.join(BASE_DIR, "data", "sample_data.jsonl")
    out_dir = os.path.join(BASE_DIR, "data")
    
    pipeline = RootCausePipeline()
    pipeline.run_pipeline(raw_file, out_dir)
