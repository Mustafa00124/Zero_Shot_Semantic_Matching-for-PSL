# Zero-Shot Semantic Matching for Pakistan Sign Language (PSL)

## Project Overview

This project tackles **isolated sign language recognition in low-data regimes** for Pakistan Sign Language (PSL). Traditional deep learning methods struggle with limited training data due to overfitting. This repository implements and compares multiple approaches designed to handle data scarcity:

- **CNN-LSTM**: Temporal modeling with frozen CNN features
- **C3D**: 3D convolutional networks with pretrained weights
- **AttentionLite-MHI**: Motion History Image guided spatial attention
- **MediaPipe**: Keypoint-based recognition with Transformer/LSTM backends
- **Zero-Shot Matching**: Structured semantic matching without training

## Quick Start

### Easiest Way to Run Experiments

The simplest way to run any method is using the **batch file** (`run_code.bat`). This allows you to:
- Test multiple methods sequentially
- Run on different data splits (train/test)
- Use multiple seeds for statistical validation

### How to Use the Batch File

1. **Open `run_code.bat`** in any text editor
2. **Configure your experiment** by modifying these lists:

```batch
set seed_list=1 42 123                    # Multiple seeds for robust results
set method_list=c3d cnn_lstm zero_shot    # Choose methods to test
set num_words_list=1 5 10                 # Number of videos to process
set split_list=train test                 # Data splits to evaluate
```

3. **Run the batch file**:
```bash
.\run_code.bat
```

### Example Configurations

**Test all methods on 1 video:**
```batch
set method_list=attentionlite_mhi c3d cnn_lstm zero_shot mediapipe_transformer mediapipe_lstm
set num_words_list=1
set split_list=train test
set seed_list=1
```

**Focus on specific methods with multiple scales:**
```batch
set method_list=c3d zero_shot
set num_words_list=1 5 10 20
set split_list=test
set seed_list=1 42 123
```

### Manual Execution

You can also run individual methods:
```bash
python main.py --method=zero_shot --split=test --num_words=5 --seed=42
```

## Data Structure

- `Words_train/`: Training videos (flat structure: all .mp4 files in one directory)
- `Words_test/`: Testing videos (flat structure: all .mp4 files in one directory)
- `results/`: Output accuracy files organized by method

## Results

Results are saved as JSON files in `results/{method}/accuracy_seed{seed}_n{num_words}_{split}.json` with format:
```json
{
  "method": "zero_shot",
  "split": "test", 
  "seed": 42,
  "num_words": 10,
  "total": 10,
  "correct": 7,
  "accuracy": 0.7
}
```

## Low-Data Challenges

In low-data regimes for sign language recognition:
- **Overfitting** is the primary challenge
- **Spatio temporal Deep learning methods** (C3D, CNN-LSTM) help with feature extraction
- **Zero-shot methods** avoid training entirely
- **Attention mechanisms** (AttentionLite-MHI) focus on motion-relevant regions
- **Keypoint-based approaches** (MediaPipe) reduce dimensionality