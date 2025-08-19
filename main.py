"""
Unified entry point that dispatches to baseline modules. No training code here.

Usage:
  python main.py --method METHOD --num_words N

Methods:
  - c3d
  - cnn_lstm
  - zero_shot
  - semantic_zero_shot
  - mhi_baseline
  - mhi_fusion
  - mhi_attention
  - finetuned_gemini
"""

import argparse
import os
import random
from methods.c3d_baseline import run_c3d
from methods.cnn_lstm_baseline import run_cnn_lstm
from methods.Mediapipe_baseline import run_mediapipe
from methods.MHI_baseline import run_mhi
from methods.finetuned_gemini_baseline import run_finetuned_gemini
from methods.zero_shot_semantic_matching import run_zero_shot_matching, run_semantic_matching

def main():
    parser = argparse.ArgumentParser(description='Run PSL video classification baselines')
    parser.add_argument('--method', 
                       choices=['c3d', 'cnn_lstm', 'mediapipe', 'mhi_baseline', 'mhi_fusion', 'mhi_attention', 
                               'finetuned_gemini', 'zero_shot', 'semantic_zero_shot'],
                       required=True,
                       help='Baseline method to run')
    parser.add_argument('--num_words', type=int, default=1, help='Number of words to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    
    # Create output directory
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Set random seed
    random.seed(args.seed)
    
    if args.method == 'c3d':
        run_c3d(num_words=args.num_words, seed=args.seed, out_dir=out_dir)
    elif args.method == 'cnn_lstm':
        run_cnn_lstm(num_words=args.num_words, seed=args.seed, out_dir=out_dir)
    elif args.method == 'mediapipe':
        run_mediapipe(num_words=args.num_words, seed=args.seed, out_dir=out_dir)
    elif args.method == 'mhi_baseline':
        run_mhi(num_words=args.num_words, mode="baseline", seed=args.seed, out_dir=out_dir)
    elif args.method == 'mhi_fusion':
        run_mhi(num_words=args.num_words, mode="fusion", seed=args.seed, out_dir=out_dir)
    elif args.method == 'mhi_attention':
        run_mhi(num_words=args.num_words, mode="attention", seed=args.seed, out_dir=out_dir)
    elif args.method == 'finetuned_gemini':
        run_finetuned_gemini(num_words=args.num_words, seed=args.seed, out_dir=out_dir)
    elif args.method == 'zero_shot':
        run_zero_shot_matching(num_words=args.num_words, seed=args.seed, out_dir=out_dir)
    elif args.method == 'semantic_zero_shot':
        run_semantic_matching(num_words=args.num_words, seed=args.seed, out_dir=out_dir)
    else:
        print(f"Unknown method: {args.method}")

if __name__ == "__main__":
    main()

