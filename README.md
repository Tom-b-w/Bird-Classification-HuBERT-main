# Bird-Classification-HuBERT 

## Quick Overview

- Environment: Python 3.8+ recommended, CUDA + PyTorch if you plan to use a GPU.
- Common dependencies: torch, transformers, datasets, librosa, scikit-learn, pandas, numpy, matplotlib, etc.
- Data: scripts commonly use paths under `./xeno_dataset/whole_dataset/...`. Check the `train_dir` / `val_dir` at the top of each script and adjust to your dataset location.

## Files and What They Do

Below is a brief description of the main scripts in this repository:

- `wavlm_xeno_large.py`
  - A full training script for single-label audio classification using Microsoft's `microsoft/wavlm-base` pretrained model.
    - Train: `./xeno_dataset/whole_dataset/train_val/train_data_backup`
    - Validation: `./xeno_dataset/whole_dataset/train_val/validation_data_backup`
  - Outputs: saves the best model (e.g. `best_wavlm_model.pth`) and final model (`wavlm_single_label_final.pth`), configuration JSON (`wavlm_large_config.json`) and training history plots.

- `wavlm.py`
  - A smaller or more compact version of the WavLM training script; useful for experimentation with fewer resources or smaller datasets. Check the top of the file for default hyperparameters and output names.
  - The `_large` variant typically uses a larger model or larger training settings (more epochs/bigger batches) for larger-scale training.

- `HuBERT.py`
  - A training script that uses HuBERT pretrained models (e.g. `facebook/hubert-*`). Architecturally similar to WavLM/wav2vec scripts but tailored to HuBERT-specific preprocessing and outputs.

- `baseline_cnn_lstm_mfcc.py`
  - A classical baseline: extract MFCC features (librosa) from audio, then train a CNN + LSTM pipeline (CNN -> LSTM -> FC) for classification.
  - Useful when data is small, resources are limited, or when comparing against pretrained-model baselines.

- `cnn_lstm_mfcc.py` / `cnn_lstm_mfcc_large.py`
  - MFCC-based CNN + LSTM implementations. The `_large` version may use a deeper network or larger batch/epoch settings.

- `cnn_lstm_attention.py` / `cnn_lstm_attention_mfcc.py`
  - Adds an attention mechanism over the CNN + LSTM backbone to weight temporal features, which can improve performance.
  - `cnn_lstm_attention_mfcc.py` is likely the MFCC-based attention variant.

> Note: Except for `wavlm_xeno_large.py`, check each script's header for exact default paths, output filenames and hyperparameters. Scripts commonly define `train_dir`, `val_dir`, `BATCH_SIZE`, `MAX_EPOCHS` near the top.


## Recommended Environment / Dependencies
Below are the recommended Python packages (you can add these to `requirements.txt` and install with `pip install -r requirements.txt`):

- torch
- torchaudio (optional)
- transformers
- datasets
- librosa
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm

Example `requirements.txt` content:

torch
transformers
datasets
librosa
numpy
pandas
scikit-learn
matplotlib
tqdm

> Note: Install `torch` using the command appropriate for your CUDA version from the official PyTorch installation guide.

## Example Data Layout
Scripts generally expect data organized by class (species) folders, for example:

xeno_dataset/
  whole_dataset/
    train_val/
      train_data_backup/
        Species_A/
          audio1.wav
          audio2.wav
        Species_B/
          ...
      validation_data_backup/
        Species_A/
        Species_B/

Each class folder contains audio files (.wav/.mp3/.flac/.ogg etc.). Scripts will walk class folders and build a list of file paths and class labels.


## How to Run (examples)
These examples assume you are in the repository root and your data is prepared. Examples use Windows PowerShell.

1) Create and activate a virtual environment

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Install dependencies (example; replace torch install command as needed):

```powershell
pip install torch transformers datasets librosa numpy pandas scikit-learn matplotlib tqdm
```

3) Run a training script (example with `wavlm_xeno_large.py`):

```powershell
python .\wavlm_xeno_large.py
```
4) Modify default paths or parameters:
   - Open the script and edit `train_dir`, `val_dir`, `BATCH_SIZE`, `MAX_EPOCHS` etc.
   - Large models (HuBERT/wav2vec/wavlm) are resource intensive; start with small batches or limited data for debugging.
