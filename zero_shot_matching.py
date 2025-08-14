#!/usr/bin/env python3
"""
test_words.py

Structured PSL test-time matcher (3-stage) using similarity thresholds.
- Stage 1: Deterministic scoring with threshold filtering
- Stage 2: Embedding similarity with threshold filtering  
- Stage 3: Gemini reasoning among all candidates above thresholds
- Outputs detailed JSON log and clean summary
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
import base64
import mimetypes
import random
import re

from sentence_transformers import SentenceTransformer, util

# ---------------------------
# Configuration
# ---------------------------
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"
MODEL_NAME = "gemini-2.0-flash"
VIDEO_DIR = "Words_test"
VOCAB_FILE = "psl_vocabulary_structured.json"
MAX_RETRIES = 3
INITIAL_DELAY = 1.0
MAX_DELAY = 10.0
TIMEOUT = 120

# Similarity thresholds (improved values)
DET_THRESHOLD = 0.55  # Lowered from 0.6 since we have fewer, more robust fields
EMB_THRESHOLD = 0.6  # Keep embedding threshold the same
EMB_THRESHOLD_FLOOR = 0.5  # Minimum threshold for adaptive fallback

# Allowed values for validation (same as vocabulary builder)
ALLOWED = {
    "hands_used": {"one", "both"},
    "primary_hand": {"left", "right", "none"},
    "two_hand_relation": {"same-action", "support-contact", "alternating", "stacked"},
    "contact_presence": {"none", "hand-hand", "hand-body"},
    "body_region_primary": {"face-head", "neck", "chest", "torso", "waist", "neutral-front"},
    "path_family": {"none", "push-pull", "open-close", "linear", "arc-circle", "twist"},
}

# Motion-spec tokens (for validation)
MOTION_TOKENS = {
    "directions": {"forward","back","up","down","left","right","inward","outward","toward-body","away-from-body"},
    "paths": {"straight","arc","circle","small-arc","big-arc","zigzag","twist","open-close","pulse"},
    "extent": {"small","medium","large"},
    "repetition": {"single","repeated"},
    "coord": {"together","mirror","parallel","counter"},
    "stability": {"static","slight-drift"},
    "targets": {"forehead","eye","cheek","chin","mouth","ear","neck","shoulder","chest","stomach","waist","neutral-front"},
}

if not API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Set it in your environment: export GEMINI_API_KEY=..."
    )

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("psl_test")

# ---------------------------
# Prompts
# ---------------------------
TEST_PROMPT = """You are given a short video of a Pakistan Sign Language (PSL) sign.
IMPORTANT: The video shows the same action performed twice - provide ONE description that covers both repetitions.

Fill the JSON exactly as specified below. Return ONLY the JSON object. No markdown, no extra text.

1) Deterministic fields (enums only — pick from allowed values; if uncertain, choose closest):
- hands_used: one | both
- primary_hand: left | right | none (use "none" if hands_used = "both")
- two_hand_relation: same-action | support-contact | alternating | stacked (use "same-action" if hands_used = "one")
- contact_presence: none | hand-hand | hand-body
- body_region_primary: face-head | neck | chest | torso | waist | neutral-front
- path_family: none | push-pull | open-close | linear | arc-circle | twist

2) Embedding fields (strict motion language). Use only these motion-spec tokens:
- Directions: forward | back | up | down | left | right | inward | outward | toward-body | away-from-body
- Paths: straight | arc | circle | small-arc | big-arc | zigzag | twist | open-close | pulse
- Extent: small | medium | large
- Repetition: single | repeated
- Coordination: together | mirror | parallel | counter
- Stability: static | slight-drift
- Targets (body refs): forehead | eye | cheek | chin | mouth | ear | neck | shoulder | chest | stomach | waist | neutral-front

Write four short lines, exactly in this order, each ≤ 12 words:
- arm_motion: (upper arms/forearms)
- hand_motion: (wrist/hand path & orientation changes)
- finger_motion: (extension/opposition/aperture)
- movement_summary: (≤12 words summary of the overall motion)

Rules:
- Use telegraphic phrases from the tokens above; join with commas.
- If still: write "static".
- If both hands same: start with "both:"; else use "left:" and "right:".

3) Reasoning-friendly fields:
- overall_description: 2–3 sentences in plain language; may use world-knowledge analogies.

EXAMPLES:

Example 1 - Pointing with both hands:
{
  "hands_used": "both",
  "primary_hand": "none",
  "two_hand_relation": "same-action",
  "contact_presence": "none",
  "body_region_primary": "chest",
  "path_family": "push-pull",
  "arm_motion": "both: small forward, repeated, together, near chest",
  "hand_motion": "both: straight pulse outward-inward, parallel",
  "finger_motion": "both: index extended, static",
  "movement_summary": "both hands point forward and back repeatedly",
  "overall_description": "Two index fingers extend from the chest, moving forward and backward in a repeated motion. The hands move together in parallel, maintaining the pointing gesture throughout."
}

Example 2 - Single hand flat gesture:
{
  "hands_used": "one",
  "primary_hand": "right",
  "two_hand_relation": "same-action",
  "contact_presence": "none",
  "body_region_primary": "chest",
  "path_family": "linear",
  "arm_motion": "right: medium down, single, toward chest",
  "hand_motion": "right: straight down, palm facing body",
  "finger_motion": "right: all extended, static",
  "movement_summary": "right flat hand moves down to chest",
  "overall_description": "A single right hand with all fingers extended moves downward from a neutral position to the chest area. The palm faces the body as the hand descends in a straight path."
}

REQUIRED JSON FORMAT:
{
  "hands_used": "one_or_both",
  "primary_hand": "left_or_right_or_none",
  "two_hand_relation": "same-action_or_support-contact_or_alternating_or_stacked",
  "contact_presence": "none_or_hand-hand_or_hand-body",
  "body_region_primary": "face-head_or_neck_or_chest_or_torso_or_waist_or_neutral-front",
  "path_family": "none_or_push-pull_or_open-close_or_linear_or_arc-circle_or_twist",
  "arm_motion": "motion_description_line",
  "hand_motion": "motion_description_line",
  "finger_motion": "motion_description_line",
  "movement_summary": "≤12_words_summary",
  "overall_description": "2-3_sentences_describing_the_visible_sign_only"
}"""

REASON_PROMPT = """You are a PSL expert. Given a test sign description and a list of candidate signs, 
choose the best match and explain why.

Consider these key factors:
1) Deterministic fields (hands_used, two_hand_relation, contact_presence, body_region_primary, path_family)
2) Motion descriptions (arm_motion, hand_motion, finger_motion, movement_summary)
3) Overall description similarity

Return ONLY a JSON object:
{
  "predicted_word": "the_best_matching_word",
  "reasoning": "explanation of why this is the best match, focusing on the most distinctive features"
}"""

# ---------------------------
# HTTP call helpers
# ---------------------------
GEN_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
HEADERS = {"Content-Type": "application/json"}

def call_gemini(prompt: str, model: str, video_path: Optional[Path] = None) -> str:
    """Call Gemini with inline video and text prompt, retries + backoff."""
    parts = [{"text": prompt}]
    
    if video_path:
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type:
            mime_type = "video/mp4"

        with open(video_path, "rb") as f:
            data = f.read()
            if len(data) > 20 * 1024 * 1024:
                raise ValueError(f"Video too large ({len(data)/1024/1024:.1f}MB): {video_path.name}")
            encoded = base64.b64encode(data).decode("utf-8")

        parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "top_p": 0.1,
            "top_k": 1
        }
    }

    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(GEN_URL, json=payload, headers=HEADERS, params={"key": API_KEY}, timeout=TIMEOUT)
            resp.raise_for_status()
            j = resp.json()
            if "error" in j:
                raise RuntimeError(j["error"].get("message", "Unknown API error"))

            cands = j.get("candidates", [])
            if not cands:
                raise RuntimeError("No candidates in response")
            parts = cands[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                raise RuntimeError("No text in response parts")
            return parts[0]["text"]

        except (requests.HTTPError, requests.Timeout) as e:
            if attempt < MAX_RETRIES - 1:
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(delay * jitter, MAX_DELAY)
                logger.warning(f"{type(e).__name__}: retrying in {sleep_time:.1f}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(sleep_time)
                delay *= 2
                continue
            raise
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(delay * jitter, MAX_DELAY)
                logger.warning(f"{type(e).__name__}: retrying in {sleep_time:.1f}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(sleep_time)
                delay *= 2
                continue
            raise

def clean_to_json_text(text: str) -> str:
    """Trim to the outermost JSON object substring."""
    text = text.strip()
    # strip markdown fences if any
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        text = text.strip()
    # find first { and last }
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return text[i:j+1]
    return text

# ---------------------------
# Validation & repair
# ---------------------------
def closest_allowed(value: str, allowed: List[str]) -> str:
    """Pick the allowed token with max character overlap (very simple)."""
    if not value:
        return next(iter(allowed))
    val = value.strip().lower()
    best = None
    best_score = -1
    for a in allowed:
        score = len(set(val) & set(a))
        if score > best_score:
            best_score = score
            best = a
    return best or next(iter(allowed))

def coerce_enum(d: Dict[str, Any], field: str, allowed: List[str]):
    """Coerce enum field to closest allowed value."""
    if field not in d:
        return
    val = d[field]
    if val in allowed:
        return
    closest = closest_allowed(val, allowed)
    d[field] = closest

def validate_motion_line(line: str) -> str:
    """
    Very lightweight sanitizer: keep known tokens, commas, colons, and spaces.
    Enforce <= 12 words.
    """
    if not isinstance(line, str):
        return "static"
    # Tokenize on non-alphanum / hyphen characters, keep commas/colons
    parts = re.split(r"([^A-Za-z0-9\-,: ])", line)
    filtered = []
    for p in parts:
        if re.fullmatch(r"[A-Za-z0-9\-]+", p):
            # only keep if it's a known token-ish or short word
            filtered.append(p)
        elif p in {",", ":", " "}:
            filtered.append(p)
        # else drop
    s = "".join(filtered)
    # limit words to 12
    words = [w for w in re.split(r"[ ,:]+", s.strip()) if w]
    if len(words) > 12:
        words = words[:12]
    return " ".join(words) if words else "static"

def ensure_list(d: Dict[str, Any], field: str):
    v = d.get(field, [])
    if not isinstance(v, list):
        d[field] = [v] if v else []

def validate_and_fix_test_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix test record to match new robust deterministic fields."""
    # Required fields for the new schema
    required_fields = [
        "hands_used", "primary_hand", "two_hand_relation", 
        "contact_presence", "body_region_primary", "path_family",
        "arm_motion", "hand_motion", "finger_motion", 
        "movement_summary", "overall_description"
    ]
    
    # Ensure all required fields exist
    for field in required_fields:
        if field not in rec:
            rec[field] = ""
    
    # Coerce enums to allowed values
    coerce_enum(rec, "hands_used", list(ALLOWED["hands_used"]))
    coerce_enum(rec, "primary_hand", list(ALLOWED["primary_hand"]))
    coerce_enum(rec, "two_hand_relation", list(ALLOWED["two_hand_relation"]))
    coerce_enum(rec, "contact_presence", list(ALLOWED["contact_presence"]))
    coerce_enum(rec, "body_region_primary", list(ALLOWED["body_region_primary"]))
    coerce_enum(rec, "path_family", list(ALLOWED["path_family"]))
    
    # Apply motion line validation
    rec["arm_motion"] = validate_motion_line(rec.get("arm_motion", "static"))
    rec["hand_motion"] = validate_motion_line(rec.get("hand_motion", "static"))
    rec["finger_motion"] = validate_motion_line(rec.get("finger_motion", "static"))
    
    # Ensure movement_summary is ≤ 12 words
    if "movement_summary" in rec:
        words = rec["movement_summary"].split()
        if len(words) > 12:
            rec["movement_summary"] = " ".join(words[:12])
    
    return rec

# ---------------------------
# Scoring functions
# ---------------------------
def deterministic_score(test: Dict[str, Any], cand: Dict[str, Any]) -> float:
    """Compute weighted deterministic similarity score between test and candidate."""
    score = 0.0
    total_weight = 0.0

    # Field weights (higher = more important)
    field_weights = {
        "path_family": 3,           # Most distinctive
        "contact_presence": 2,      # Very distinctive
        "hands_used": 2,            # Important structural feature
        "two_hand_relation": 2,     # Important structural feature
        "body_region_primary": 1,   # Less critical but still useful
        "primary_hand": 1,          # Least critical
    }

    # Compare deterministic fields with weights
    for field, weight in field_weights.items():
        if field in test and field in cand:
            total_weight += weight
            if test[field] == cand[field]:
                score += weight
            # Special case: primary_hand mismatch but both hands (common LLM bias)
            elif field == "primary_hand" and test.get("hands_used") == "both" and cand.get("hands_used") == "both":
                # Small penalty for primary_hand mismatch when using both hands
                score += weight * 0.7

    # Bonus for matching hands_used + two_hand_relation (both hands)
    if (test.get("hands_used") == "both" and cand.get("hands_used") == "both" and
        test.get("two_hand_relation") == cand.get("two_hand_relation")):
        score += 1.0  # Bonus for perfect hand relation match

    return score / total_weight if total_weight > 0 else 0.0

def deterministic_shortlist(test_rec: Dict[str, Any], candidates: List[Dict[str, Any]], threshold: float = 0.55) -> List[Tuple[Dict[str, Any], float]]:
    """Get all candidates above deterministic threshold, re-ranked by weight."""
    scored = []
    for c in candidates:
        s = deterministic_score(test_rec, c)
        if s >= threshold:
            scored.append((c, s))
    
    # Re-rank by score (higher is better)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def motion_text(rec: Dict[str, Any]) -> str:
    """Extract motion description text for embedding."""
    return "; ".join([
        rec.get("arm_motion",""),
        rec.get("hand_motion",""),
        rec.get("finger_motion",""),
        rec.get("movement_summary",""),
    ]).strip()

def embedding_shortlist(test_rec: Dict[str, Any], det_list: List[Tuple[Dict[str, Any], float]], model: SentenceTransformer, threshold: float = 0.6) -> List[Tuple[Dict[str, Any], float, float]]:
    """Get candidates above embedding similarity threshold with adaptive fallback."""
    if not det_list:
        return []
        
    test_text = motion_text(test_rec)
    test_emb = model.encode(test_text, convert_to_tensor=True, normalize_embeddings=True)

    c_texts = [motion_text(c[0]) for c in det_list]
    c_embs = model.encode(c_texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(test_emb, c_embs).cpu().tolist()[0]

    combined = []
    for (cand, det_score), sim in zip(det_list, sims):
        if sim >= threshold:
            combined.append((cand, float(sim), det_score))
    
    # If no candidates above threshold, try adaptive threshold
    if not combined and threshold > EMB_THRESHOLD_FLOOR:
        logger.info(f"No embedding candidates above {threshold}, trying adaptive threshold...")
        adaptive_threshold = threshold - 0.05
        for (cand, det_score), sim in zip(det_list, sims):
            if sim >= adaptive_threshold:
                combined.append((cand, float(sim), det_score))
        if combined:
            logger.info(f"Found {len(combined)} candidates with adaptive threshold {adaptive_threshold}")
    
    # If still no candidates, fallback to top-k deterministic
    if not combined:
        logger.info(f"No embedding candidates found, falling back to top-5 deterministic")
        top_k = min(5, len(det_list))
        for i in range(top_k):
            cand, det_score = det_list[i]
            # Use deterministic score as similarity for fallback
            combined.append((cand, det_score, det_score))
    
    combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return combined

# ---------------------------
# Reasoning stage
# ---------------------------
def format_shortlist_for_prompt(shortlist: List[Dict[str, Any]]) -> str:
    """Format shortlist for Gemini prompt with correct fields for reasoner."""
    slim = []
    for c in shortlist:
        slim.append({
            "word": c["word"],
            "hands_used": c["hands_used"],
            "primary_hand": c["primary_hand"],
            "two_hand_relation": c["two_hand_relation"],
            "contact_presence": c["contact_presence"],
            "body_region_primary": c["body_region_primary"],
            "path_family": c["path_family"],
            "arm_motion": c["arm_motion"],
            "hand_motion": c["hand_motion"],
            "finger_motion": c["finger_motion"],
            "movement_summary": c.get("movement_summary", ""),
            "overall_description": c.get("overall_description", "")
        })
    return json.dumps(slim, ensure_ascii=False, indent=2)

def reason_select(test_rec: Dict[str, Any], shortlist: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Use Gemini to reason and select best match from shortlist."""
    if not shortlist:
        return "", "No candidates above similarity thresholds"
    
    test_json = json.dumps(test_rec, ensure_ascii=False, indent=2)
    cand_json = format_shortlist_for_prompt(shortlist)
    prompt = f"""{REASON_PROMPT}

TEST:
{test_json}

SHORTLIST (each item has a 'word' you must choose from):
{cand_json}
"""
    raw = call_gemini(prompt, MODEL_NAME, video_path=None)
    txt = clean_to_json_text(raw)
    try:
        j = json.loads(txt)
        return j.get("predicted_word",""), j.get("reasoning","")
    except json.JSONDecodeError:
        logger.warning("Reasoner JSON parse failed. Falling back to top embedding candidate.")
        return shortlist[0]["word"], "Fallback to top embedding candidate due to parse error."

# ---------------------------
# I/O
# ---------------------------
def load_vocabulary(path: str) -> List[Dict[str, Any]]:
    """Load vocabulary from JSON file."""
    try:
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        # ensure required fields exist
        for v in vocab:
            for f in ["visual_analogies"]:
                v.setdefault(f, [])
        return vocab
    except Exception as e:
        logger.error(f"Failed to load vocabulary: {e}")
        return []

def describe_test_video(video_path: Path) -> Dict[str, Any]:
    """Get structured description of test video using Gemini."""
    logger.info(f"Describing test video: {video_path.name}")
    raw = call_gemini(TEST_PROMPT, MODEL_NAME, video_path=video_path)
    txt = clean_to_json_text(raw)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        logger.error(f"JSON parse failed for {video_path.name}. Saving raw.")
        Path("debug").mkdir(exist_ok=True)
        (Path("debug") / f"{video_path.stem}_raw.txt").write_text(raw, encoding="utf-8")
        raise

    rec = validate_and_fix_test_record(data)
    return rec

# ---------------------------
# Main testing function
# ---------------------------
def test_words(num_words: int, video_dir: str = VIDEO_DIR, seed: int = 42, out_dir: str = "results"):
    """Test PSL matching and save accuracy and detailed logs under results/zero_shot."""
    vocab = load_vocabulary(VOCAB_FILE)
    if not vocab:
        logger.error("Empty or invalid vocabulary file.")
        return

    # Load embedding model once
    logger.info("Loading sentence-transformers: all-MiniLM-L6-v2")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    test_dir = Path(video_dir)
    vids = sorted(list(test_dir.glob("*.mp4")))
    if not vids:
        logger.error(f"No .mp4 files found in {video_dir}")
        return

    # Seed selection
    import random
    rng = random.Random(seed)
    rng.shuffle(vids)
    vids = vids[:min(num_words, len(vids))]
    results = []
    detailed_results = []

    for i, vp in enumerate(vids, 1):
        actual = vp.stem  # ground-truth from filename
        logger.info(f"\n({i}/{len(vids)}) {vp.name}")

        try:
            # 1) Structured test record
            t0 = time.time()
            test_rec = describe_test_video(vp)

            # 2) Deterministic shortlist (threshold-based)
            det = deterministic_shortlist(test_rec, vocab, threshold=DET_THRESHOLD)
            logger.info(f"Deterministic candidates: {len(det)} (threshold: {DET_THRESHOLD})")

            # 3) Embedding shortlist (threshold-based with fallback)
            emb_short = embedding_shortlist(test_rec, det, st_model, threshold=EMB_THRESHOLD)
            cand_list = [c for (c, sim, dsc) in emb_short]
            logger.info(f"Embedding candidates: {len(cand_list)} (threshold: {EMB_THRESHOLD})")
            
            # Log if fallback was used
            if len(cand_list) > 0 and len(det) > 0:
                first_cand = emb_short[0]
                if first_cand[1] == first_cand[2]:  # sim == det_score indicates fallback
                    logger.info("⚠️  Using deterministic fallback (no embedding candidates above threshold)")

            # 4) Reasoning select
            pred_word, reasoning = reason_select(test_rec, cand_list)

            elapsed = round(time.time() - t0, 2)
            
            # Find the matched vocab entry
            matched = next((c for c in cand_list if c["word"] == pred_word), None)

            # Basic result for summary
            result = {
                "video_file": vp.name,
                "actual_word": actual,
                "predicted_word": pred_word,
                "correct": (pred_word == actual),
                "elapsed_sec": elapsed
            }
            results.append(result)

            # Detailed result for logging
            detailed_result = {
                "video_file": vp.name,
                "actual_word": actual,
                "test_record": test_rec,
                "deterministic_candidates": [(c["word"], score) for c, score in det],
                "embedding_candidates": [(c["word"], sim, det_score) for c, sim, det_score in emb_short],
                "predicted_word": pred_word,
                "reasoning": reasoning,
                "matched_entry": matched,
                "elapsed_sec": elapsed
            }
            detailed_results.append(detailed_result)

            ok = (pred_word == actual)
            logger.info(f"Predicted: {pred_word} | Actual: {actual} | {'✅' if ok else '❌'} | {elapsed}s")

            # Gentle pacing
            time.sleep(2)

        except Exception as e:
            logger.exception(f"Failed on {vp.name}: {e}")

    # Summary
    correct = sum(1 for r in results if r["correct"])
    acc = (correct / len(results) * 100) if results else 0.0
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tested: {len(results)}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Accuracy: {acc:.1f}%")

    # Save under results/zero_shot
    method_dir = Path(out_dir) / "zero_shot"
    method_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Simple accuracy file
    acc_path = method_dir / f"accuracy_seed{seed}_n{num_words}.json"
    acc_payload = {
        "timestamp": ts,
        "method": "zero_shot",
        "seed": seed,
        "num_words": num_words,
        "total_tested": len(results),
        "correct": correct,
        "accuracy": acc,
    }
    acc_path.write_text(json.dumps(acc_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved accuracy → {acc_path}")

    # Detailed file with logs of descriptions and candidates
    detailed_path = method_dir / f"detailed_seed{seed}_n{num_words}.json"
    detailed_payload = {
        "timestamp": ts,
        "thresholds": {
            "deterministic": DET_THRESHOLD,
            "embedding": EMB_THRESHOLD,
            "embedding_floor": EMB_THRESHOLD_FLOOR
        },
        "results": detailed_results
    }
    detailed_path.write_text(json.dumps(detailed_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved detailed logs → {detailed_path}")

# ------------------------------------
# CLI
# ------------------------------------
def run_zero_shot(num_words: int = 1, video_dir: str = VIDEO_DIR, seed: int = 42, out_dir: str = "results"):
    """Convenience wrapper to match other baselines."""
    return test_words(num_words, video_dir=video_dir, seed=seed, out_dir=out_dir)
def main():
    ap = argparse.ArgumentParser(description="Structured PSL test-time matcher (3-stage) with similarity thresholds")
    ap.add_argument("--num_words", type=int, default=1, help="How many test videos to run.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    test_words(args.num_words, video_dir=args.video_dir, seed=args.seed, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
