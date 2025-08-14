"""
Unified entry point that dispatches to baseline modules. No training code here.

Usage:
  python main.py --method METHOD --num_words N [--split train|test]

Methods:
  - attentionlite_mhi
  - c3d
  - cnn_lstm
  - zero_shot
  - mediapipe_transformer
  - mediapipe_lstm
"""

import argparse

from attentionlite_MHI_baseline import run_attentionlite_mhi
from c3d_baseline import run_c3d
from cnn_lstm_baseline import run_cnn_lstm
from zero_shot_matching import run_zero_shot
from Mediapipe_baseline import run_mediapipe


def main():
    parser = argparse.ArgumentParser(description="Unified PSL Sign Recognition Runner")
    parser.add_argument('--method', type=str, required=True,
                        choices=['attentionlite_mhi', 'c3d', 'cnn_lstm', 'zero_shot', 'mediapipe_transformer', 'mediapipe_lstm'],
                        help='Method to run')
    parser.add_argument('--num_words', type=int, default=1,
                        help='Number of samples/videos to run')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Choose dataset split for testing: train -> Words_train, test -> Words_test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='results')

    args = parser.parse_args()

    if args.method == 'attentionlite_mhi':
        run_attentionlite_mhi(num_words=args.num_words, split=args.split, seed=args.seed, out_dir=args.out_dir)
    elif args.method == 'c3d':
        run_c3d(num_words=args.num_words, split=args.split, seed=args.seed, out_dir=args.out_dir)
    elif args.method == 'cnn_lstm':
        run_cnn_lstm(num_words=args.num_words, split=args.split, seed=args.seed, out_dir=args.out_dir)
    elif args.method == 'zero_shot':
        run_zero_shot(num_words=args.num_words, split=args.split, seed=args.seed, out_dir=args.out_dir)
    elif args.method == 'mediapipe_transformer':
        run_mediapipe(num_words=args.num_words, split=args.split, backend='transformer', seed=args.seed, out_dir=args.out_dir)
    elif args.method == 'mediapipe_lstm':
        run_mediapipe(num_words=args.num_words, split=args.split, backend='lstm', seed=args.seed, out_dir=args.out_dir)

if __name__ == '__main__':
    main()

