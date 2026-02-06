# Emotion Detection

A deep learning project for detecting emotions from facial expressions using Convolutional Neural Networks (CNN).

## Features

- **Static Image Detection**: Analyze emotions from images
- **Realtime Detection**: Live emotion detection using webcam
- 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Requirements

- Python 3.8+
- Dependencies listed in `pyproject.toml`

## Installation

Install dependencies using pip:

```bash
pip install -e .
```

Or using uv:

```bash
uv sync
```

## Usage

### Realtime Emotion Detection (Webcam)

Run the realtime detection script:

```bash
python realtime.py
```

- The script will open your webcam and detect emotions in realtime
- Press `q` or `ESC` to quit

### Static Image Detection

Edit the `IMAGE_PATH` in `main.py` to point to your image, then run:

```bash
python main.py
```

## Model

The project uses a pre-trained model saved as `best_emotion_model.keras`. The model is trained on facial emotion recognition dataset with 7 emotion categories.

## Training

To train your own model, use `modeling.py`:

```bash
python modeling.py
```

Make sure you have the dataset folder structured as:
```
dataset/
  train/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
  test/
    angry/
    ...
```

## Project Structure

- `realtime.py` - Realtime emotion detection using webcam
- `main.py` - Static image emotion detection
- `modeling.py` - Model training script
- `best_emotion_model.keras` - Pre-trained model
- `dataset/` - Training dataset directory
