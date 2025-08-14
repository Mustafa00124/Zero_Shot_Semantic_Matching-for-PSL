#!/usr/bin/env python3
"""
make_structured_dictionary.py

Builds a structured PSL vocabulary from sign videos using Gemini.
- Uses a strict, enum-driven prompt for deterministic fields
- Captures motion-spec lines for embedding
- Adds reasoning-friendly visual analogies and rationale
- Validates/repairs model outputs to the allowed schema
"""

import os
import requests
import json
from pathlib import Path
import time
import mimetypes
import base64
import random
import re
import logging
import argparse
from typing import Any, Dict, List, Optional

# ---------------------------
# Configuration
# ---------------------------
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"
MODEL_NAME = "gemini-2.0-flash"
VIDEO_DIR = "Words"
OUTPUT_JSON = "psl_vocabulary_structured.json"
MAX_RETRIES = 5
INITIAL_DELAY = 1.0
MAX_DELAY = 30.0
TIMEOUT = 180

if not API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Set it in your environment: export GEMINI_API_KEY=..."
    )

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("psl_dict")

# ---------------------------
# Allowed values (deterministic fields)
# ---------------------------
ALLOWED = {
    "hands_used": {"one", "both"},
    "primary_hand": {"left", "right", "none"},
    "two_hand_relation": {"same-action", "support-contact", "alternating", "stacked"},
    "contact_presence": {"none", "hand-hand", "hand-body"},
    "body_region_primary": {"face-head", "neck", "chest", "torso", "waist", "neutral-front"},
    "path_family": {"none", "push-pull", "open-close", "linear", "arc-circle", "twist"},
}

# Motion-spec tokens (for quick validation/cleanup if needed)
MOTION_TOKENS = {
    "directions": {"forward","back","up","down","left","right","inward","outward","toward-body","away-from-body"},
    "paths": {"straight","arc","circle","small-arc","big-arc","zigzag","twist","open-close","pulse"},
    "extent": {"small","medium","large"},
    "repetition": {"single","repeated"},
    "coord": {"together","mirror","parallel","counter"},
    "stability": {"static","slight-drift"},
    "targets": {"forehead","eye","cheek","chin","mouth","ear","neck","shoulder","chest","stomach","waist","neutral-front"},
}

# ---------------------------
# Prompt (strict)
# ---------------------------
PROMPT_TEMPLATE = """
You are given a short video of a Pakistan Sign Language (PSL) sign.
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
- head_motion: (head movement if significant - specify significance: "minimal", "moderate", or "significant"; if none, write "none")

Rules:
- Use telegraphic phrases from the tokens above; join with commas.
- If still: write "static".
- If both hands same: start with "both:"; else use "left:" and "right:".
- For head_motion: only include if there's visible head movement, and specify how significant it appears.

3) Reasoning-friendly fields:
- overall_description: 2–3 sentences in plain language; may use world-knowledge analogies.
- movement_summary: ≤ 12 words (crisp phrase of the motion).
- analogy_description: 1–2 sentences that describe what the sign looks like using visual metaphors (e.g., shapes, everyday actions, glyph-like forms). Write freely.
- visual_rationale: 1 sentence naming the most distinguishing cue.

REQUIRED JSON FORMAT:
{
  "word": "%(word)s",
  "hands_used": "both",
  "primary_hand": "none",
  "two_hand_relation": "same-action",
  "contact_presence": "none",
  "body_region_primary": "chest",
  "path_family": "push-pull",

  "arm_motion": "both: small forward, repeated, together, near chest",
  "hand_motion": "both: straight pulse outward-inward, parallel",
  "finger_motion": "both: index extended, static",
  "head_motion": "none",

  "overall_description": "Two–three sentences describing the visible sign only.",
  "movement_summary": "≤12 words summary",
  "analogy_description": "Like gently pushing and releasing two small drawers. The two pointers move in a tiny push–pull.",
  "visual_rationale": "One sentence on the key distinguishing cue."
}

Validation rules:
- All fields required. If unknown, choose the closest allowed value; never leave blank.
- Return only the JSON.
"""

# ---------------------------
# HTTP call helpers
# ---------------------------
GEN_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
HEADERS = {"Content-Type": "application/json"}

def call_gemini(prompt: str, video_path: Path) -> str:
    """Call Gemini with inline video and text prompt, retries + backoff."""
    parts = [{"text": prompt}]
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
REQUIRED_FIELDS = [
    "word","hands_used","primary_hand","two_hand_relation",
    "contact_presence","body_region_primary","path_family",
    "arm_motion","hand_motion","finger_motion","head_motion",
    "overall_description","movement_summary",
    "analogy_description","visual_rationale"
]

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

def coerce_enum(d: Dict[str, Any], field: str, allowed: set):
    v = d.get(field, "")
    if isinstance(v, str):
        v_norm = v.strip()
        if v_norm in allowed:
            d[field] = v_norm
        else:
            d[field] = closest_allowed(v_norm.lower(), sorted(list(allowed)))
    else:
        d[field] = next(iter(allowed))

def coerce_list_enum(d: Dict[str, Any], field: str, allowed: set, min_n=1, max_n=3):
    v = d.get(field, [])
    if not isinstance(v, list):
        v = [str(v)] if v else []
    out = []
    for item in v:
        token = str(item).strip()
        if token in allowed:
            out.append(token)
        else:
            out.append(closest_allowed(token.lower(), sorted(list(allowed))))
        if len(out) >= max_n:
            break
    if len(out) < min_n:
        out = ["none"]
    d[field] = out

def ensure_list(d: Dict[str, Any], field: str):
    v = d.get(field, [])
    if not isinstance(v, list):
        d[field] = [v] if v else []

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

def validate_and_fix_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure all required fields exist
    for f in REQUIRED_FIELDS:
        rec.setdefault(f, "" if f not in {"analogy_description"} else [])

    # Coerce enums
    coerce_enum(rec, "hands_used", ALLOWED["hands_used"])
    coerce_enum(rec, "primary_hand", ALLOWED["primary_hand"])
    coerce_enum(rec, "two_hand_relation", ALLOWED["two_hand_relation"])
    coerce_enum(rec, "contact_presence", ALLOWED["contact_presence"])
    coerce_enum(rec, "body_region_primary", ALLOWED["body_region_primary"])
    coerce_enum(rec, "path_family", ALLOWED["path_family"])

    # Lists
    ensure_list(rec, "analogy_description")

    # Motion lines (light sanitize)
    rec["arm_motion"] = validate_motion_line(rec.get("arm_motion", "static"))
    rec["hand_motion"] = validate_motion_line(rec.get("hand_motion", "static"))
    rec["finger_motion"] = validate_motion_line(rec.get("finger_motion", "static"))
    rec["head_motion"] = validate_motion_line(rec.get("head_motion", "none"))

    # Trim text fields
    for tf in ["overall_description","movement_summary","visual_rationale","word"]:
        v = rec.get(tf, "")
        if not isinstance(v, str):
            v = str(v)
        rec[tf] = v.strip()

    # movement_summary ≤ 12 words
    words = rec["movement_summary"].split()
    if len(words) > 12:
        rec["movement_summary"] = " ".join(words[:12])

    return rec

# ---------------------------
# Main processing
# ---------------------------
def process_video(word: str, video_path: Path) -> Optional[Dict[str, Any]]:
    prompt = PROMPT_TEMPLATE % {"word": word}
    raw = call_gemini(prompt, video_path)
    json_text = clean_to_json_text(raw)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        logger.error(f"JSON parse failed for {video_path.name}. Saving raw.")
        Path("debug").mkdir(exist_ok=True)
        (Path("debug") / f"{video_path.stem}_raw.txt").write_text(raw, encoding="utf-8")
        return None

    rec = validate_and_fix_record(data)
    return rec

def generate_vocabulary(limit: Optional[int]) -> List[Dict[str, Any]]:
    video_dir = Path(VIDEO_DIR)
    if not video_dir.exists():
        logger.error(f"Directory not found: {video_dir}")
        return []

    files = [p for p in video_dir.glob("*.*") if p.suffix.lower() in {".mp4",".mov",".avi"}]
    files.sort()
    if limit is not None and limit != -1:
        files = files[:limit]

    logger.info(f"Found {len(files)} video files.")
    out: List[Dict[str, Any]] = []

    for i, vp in enumerate(files, 1):
        word = vp.stem
        logger.info(f"({i}/{len(files)}) {word}")
        t0 = time.time()
        try:
            rec = process_video(word, vp)
            if rec:
                rec["video_file"] = vp.name
                out.append(rec)
                logger.info(f"OK: {word}")
            else:
                logger.warning(f"Skipped: {word}")
        except Exception as e:
            logger.exception(f"Error on {vp.name}: {e}")

        time.sleep(2)  # gentle pacing

    return out

def main():
    ap = argparse.ArgumentParser(description="Generate structured PSL vocabulary from videos")
    ap.add_argument("--num_words", type=int, default=1, help="How many videos to process.")
    args = ap.parse_args()

    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    entries = generate_vocabulary(args.num_words)

    if entries:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(entries)} entries → {OUTPUT_JSON}")
    else:
        logger.error("No entries generated.")

if __name__ == "__main__":
    main()
