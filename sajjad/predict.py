
import json
from pathlib import Path
import time
import sys
import base64
import mimetypes
import requests
import re
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
API_KEY = "AIzaSyBNsavVonNcYikKM0hTzPwtehVmFjPLJZo"  # Replace with actual key
VIDEO_MODEL = "gemini-2.0-flash"
VOCAB_BANK = "psl_vocabulary_bank.json"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Strict structured prompt template with JSON output
STRUCTURED_PROMPT = (
    "Generate EXCLUSIVELY a JSON array describing this sign language video with these REQUIREMENTS:\n"
    "1. Break into 10 fixed 0.1-second segments from 0.0s to 1.0s\n"
    "2. For EACH segment include ONLY these EXACT keys in order:\n"
    "   - hand_shape (ONLY: 'open/relaxed', 'fist', 'index-pointed', 'flat-hand', 'claw')\n"
    "   - location (ONLY: 'at-sides', 'near-chin', 'near-face', 'chest-level', 'below-chest')\n"
    "   - movement (ONLY: 'stationary', 'upward', 'downward', 'circular', 'side-to-side')\n"
    "   - dominant_hand (ONLY: 'right', 'left', 'both', 'N/A')\n"
    "   - facial_expression (ONLY: 'neutral', 'smiling', 'raised-eyebrows')\n"
    "3. Use EXACT timing boundaries:\n"
    "   - Segment 1: 0.0s-0.1s\n"
    "   - Segment 2: 0.1s-0.2s\n"
    "   - Segment 3: 0.2s-0.3s\n"
    "   - Segment 4: 0.3s-0.4s\n"
    "   - Segment 5: 0.4s-0.5s\n"
    "   - Segment 6: 0.5s-0.6s\n"
    "   - Segment 7: 0.6s-0.7s\n"
    "   - Segment 8: 0.7s-0.8s\n"
    "   - Segment 9: 0.8s-0.9s\n"
    "   - Segment 10: 0.9s-1.0s\n\n"
    "OUTPUT FORMAT:\n"
    "[{\"segment\": 1, \"start\": 0.0, \"end\": 0.1, \"hand_shape\": ...}, ...]\n\n"
    "EXAMPLE OUTPUT:\n"
    """[
      {
        "segment": 1,
        "start": 0.0,
        "end": 0.1,
        "hand_shape": "open/relaxed",
        "location": "at-sides",
        "movement": "stationary",
        "dominant_hand": "N/A",
        "facial_expression": "neutral"
      },
      {
        "segment": 2,
        "start": 0.1,
        "end": 0.2,
        "hand_shape": "flat-hand",
        "location": "chest-level",
        "movement": "upward",
        "dominant_hand": "right",
        "facial_expression": "neutral"
      }
    ]"""
)

def describe_sign_from_video(video_path: Path) -> list:
    """Generates structured video description with enforced consistency"""
    print(f"📹 Analyzing video: {video_path.name}")
    
    if not API_KEY or "YOUR_GEMINI_API_KEY" in API_KEY:
        print(" ERROR: Missing valid Gemini API key")
        return None

    try:
        # Encode video
        mime_type, _ = mimetypes.guess_type(video_path) or "video/mp4"
        with open(video_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode("utf-8")

        # Build payload with strict prompt and JSON enforcement
        payload = {
            "contents": [{
                "parts": [
                    {"text": STRUCTURED_PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": encoded_video}}
                ]
            }],
            "generationConfig": {
                "temperature": 0,  # Critical for consistency
                "maxOutputTokens": 2000,
                "response_mime_type": "application/json"  # Force JSON output
            }
        }

        # API request
        response = requests.post(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{VIDEO_MODEL}:generateContent?key={API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        
        # Extract and parse JSON response
        response_json = response.json()
        content = response_json["candidates"][0]["content"]["parts"][0]["text"]
        
        # Clean JSON response if wrapped in markdown
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        
        return json.loads(content)
        
    except Exception as e:
        print(f" API Error: {str(e)[:200]}")
        if 'response' in locals() and response.status_code != 200:
            print(f"Response: {response.text[:300]}")
        return None

def segments_to_text(segments: list) -> str:
    """Convert structured segments to consistent text format for embedding"""
    text = ""
    for seg in segments:
        text += f"Segment {seg['segment']} ({seg['start']}s-{seg['end']}s):\n"
        text += f"- Hand Shape: {seg['hand_shape']}\n"
        text += f"- Location: {seg['location']}\n"
        text += f"- Movement: {seg['movement']}\n"
        text += f"- Dominant Hand: {seg['dominant_hand']}\n"
        text += f"- Facial Expression: {seg['facial_expression']}\n\n"
    return text.strip()

def predict_sign(test_video_path: Path, vocab_data: list):
    """Full prediction pipeline with strict formatting"""
    # Generate new description
    test_segments = describe_sign_from_video(test_video_path)
    if not test_segments:
        return None
    
    # Convert to consistent text format
    test_description = segments_to_text(test_segments)
    
    # Semantic comparison
    print(" Comparing with vocabulary bank...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Prepare vocabulary descriptions
    vocab_descriptions = []
    for entry in vocab_data:
        if isinstance(entry['description'], list):
            # New structured format
            vocab_descriptions.append(segments_to_text(entry['description']))
        else:
            # Legacy string format
            vocab_descriptions.append(entry['description'])
    
    # Generate embeddings
    test_embedding = model.encode(test_description, convert_to_tensor=True)
    vocab_embeddings = model.encode(vocab_descriptions, convert_to_tensor=True)

    # Find best match
    cosine_scores = util.cos_sim(test_embedding, vocab_embeddings)
    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[0][best_idx].item()
    
    return {
        "predicted_word": vocab_data[best_idx]['word'],
        "confidence_score": f"{best_score:.4f}",
        "input_video": test_video_path.name,
        "generated_segments": test_segments,  # Store structured data
        "text_description": test_description  # Store text version
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_from_video.py <path_to_your_video.mp4>")
        return
        
    test_video_path = Path(sys.argv[1])
    if not test_video_path.exists():
        print(f" Test video not found: {test_video_path}")
        return

    try:
        with open(VOCAB_BANK, "r") as f:
            vocab_data = json.load(f)
        print(f" Loaded vocabulary bank with {len(vocab_data)} entries.")
    except Exception as e:
        print(f" Error loading or parsing '{VOCAB_BANK}': {e}")
        return

    start_time = time.time()
    result = predict_sign(test_video_path, vocab_data)
    
    if result:
        print("\n" + "="*50)
        print(" PREDICTION RESULT")
        print(f"Predicted Word:     {result.get('predicted_word', 'N/A')}")
        print(f"Confidence Score:   {result.get('confidence_score', 'N/A')}")
        print(f"Input Video:        {result.get('input_video', 'N/A')}")
        print("="*50)
        print(f"\n Total time: {time.time() - start_time:.2f} seconds")

        output_file = f"prediction_{test_video_path.stem}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved full results to '{output_file}'")
    else:
        print("\n Prediction failed.")

if __name__ == "__main__":
    main()