# Speech Emotion Recognition Using Deep Learning

A Speech Emotion Recognition (SER) system that identifies human emotions from audio recordings using a hybrid **1D-CNN + Bidirectional LSTM** deep learning architecture. The project combines four publicly available speech emotion datasets to build a multi-dataset emotion classification model.

## Project Overview

Speech Emotion Recognition is a machine learning task that identifies the emotional state of a speaker by analyzing characteristics of their voice.

This project processes audio recordings, extracts acoustic features, applies data augmentation, and uses a deep learning model to classify speech into seven emotion categories.

### Key Features

- Integration of four speech emotion datasets.
- Audio preprocessing and feature extraction using Librosa.
- Data augmentation through noise injection and pitch shifting.
- Hybrid 1D-CNN and Bidirectional LSTM architecture.
- Seven-class emotion classification.
- Model training and evaluation using TensorFlow/Keras.

## Supported Emotions

The model classifies speech into the following seven categories:

| Emotion | Description |
|---|---|
| Neutral | Normal emotional speech |
| Happy | Positive and cheerful speech |
| Sad | Low-energy, sorrowful speech |
| Angry | Aggressive or frustrated speech |
| Fear | Anxious or fearful speech |
| Disgust | Speech expressing dislike |
| Surprise | Unexpected emotional reaction |

## Datasets Used

The project integrates four publicly available datasets:

| Dataset | Description |
|---|---|
| RAVDESS | Ryerson Audio-Visual Database of Emotional Speech and Song |
| CREMA-D | Crowd-sourced Emotional Multimodal Actors Dataset |
| TESS | Toronto Emotional Speech Set |
| SAVEE | Surrey Audio-Visual Expressed Emotion |

All datasets are mapped into a common seven-emotion classification structure.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Audio Processing:** Librosa, SoundFile
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Progress Monitoring:** tqdm
- **Model Serialization:** Joblib

## System Architecture

The project follows this pipeline:

1. **Dataset Collection:** Download audio recordings from RAVDESS, CREMA-D, TESS, and SAVEE.
2. **Dataset Integration:** Extract emotion labels from filenames and standardize them across datasets.
3. **Audio Preprocessing:** Load audio at 22,050 Hz and process recordings up to 3.5 seconds.
4. **Feature Extraction:** Extract MFCC, Chroma, and Mel Spectrogram features.
5. **Data Augmentation:** Generate additional samples using noise injection and pitch shifting.
6. **Feature Scaling:** Apply StandardScaler to normalize extracted features.
7. **Model Training:** Train a hybrid CNN-BiLSTM model.
8. **Evaluation:** Measure classification performance using accuracy and loss.

## Audio Feature Extraction

Three acoustic feature representations are extracted from every audio recording.

| Feature | Dimensions | Purpose |
|---|---:|---|
| MFCC | 40 | Captures vocal timbre and spectral characteristics |
| Chroma | 12 | Represents energy across 12 pitch classes |
| Mel Spectrogram | 128 | Captures energy distribution across Mel frequency bands |
| **Total** | **180** | Combined feature vector |

The three feature vectors are concatenated into a single 180-dimensional representation.

### Data Augmentation

To improve model generalization, two augmentation techniques are applied:

- **Noise Injection:** Adds random Gaussian noise to the original audio.
- **Pitch Shifting:** Changes the pitch by randomly shifting it by +1.5 or -1.5 semitones.

Each recording can generate up to three samples: the original audio and two augmented versions.

## Deep Learning Model

The project uses a hybrid **1D-CNN + Bidirectional LSTM** architecture.

### Model Architecture

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 180 × 1 | Accepts extracted features |
| Conv1D | 128 filters, kernel size 5 | Extracts local patterns |
| Batch Normalization | — | Stabilizes training |
| MaxPooling1D | Pool size 2 | Reduces feature dimensions |
| Dropout | 0.20 | Reduces overfitting |
| Conv1D | 256 filters, kernel size 5 | Learns deeper patterns |
| Batch Normalization | — | Normalizes activations |
| MaxPooling1D | Pool size 2 | Reduces sequence length |
| Dropout | 0.25 | Regularization |
| Bidirectional LSTM | 128 units | Learns forward and backward dependencies |
| Dropout | 0.30 | Regularization |
| Dense | 128 neurons, ReLU | Learns high-level representations |
| Batch Normalization | — | Stabilizes activations |
| Dropout | 0.30 | Regularization |
| Output | 7 neurons, Softmax | Predicts emotion probabilities |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 64 |
| Maximum Epochs | 45 |
| Train-Test Split | 85:15 |
| Random State | 42 |
| Early Stopping Patience | 8 |
| Learning Rate Reduction Factor | 0.5 |

The model uses EarlyStopping to prevent unnecessary training and ReduceLROnPlateau to lower the learning rate when validation loss stops improving.

## Model Performance

The model was trained for 45 epochs using the combined dataset.

<details>
<summary>Training Results</summary>

| Metric | Result |
|---|---:|
| Final Training Accuracy | 85.40% |
| Final Validation Accuracy | 78.97% |
| Final Test Accuracy (reported) | **78.81%** |
| Final Training Loss | 0.3893 |
| Final Validation Loss | 0.5873 |
| Final Learning Rate | 0.00003125 |

</details>

### Training Progress

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 46.86% | 59.06% |
| 5 | 64.05% | 66.43% |
| 10 | 69.02% | 71.30% |
| 15 | 72.96% | 73.68% |
| 20 | 77.14% | 74.95% |
| 25 | 79.98% | 76.75% |
| 30 | 82.49% | 78.16% |
| 35 | 83.99% | 78.62% |
| 40 | 84.93% | 78.70% |
| 45 | 85.40% | 78.97% |

The model shows consistent improvement in training and validation accuracy throughout training.

**Note:** The reported 78.81% is the evaluation accuracy from the supplied training run. The same data was used for validation during training and final evaluation, so it should not be described as performance on an independent test set. A separate untouched test set is recommended for reliable evaluation.

## Future Improvements

- Use speaker-independent train, validation, and test splits.
- Preserve frame-level audio features to model temporal information more effectively.
- Experiment with log-Mel spectrograms and attention mechanisms.
- Evaluate performance using precision, recall, F1-score, and confusion matrices.
- Explore real-time emotion prediction through a Streamlit interface.
- Improve generalization across different speakers and recording environments.

## Limitations

- Emotion recognition can be affected by speaker differences, background noise, and recording quality.
- Vocal emotion does not always represent a person's actual emotional state.
- The current implementation uses time-averaged acoustic features, meaning the BiLSTM does not receive the original chronological audio frames.
- Randomly splitting augmented samples can cause data leakage if augmented versions of the same recording appear in both training and validation sets. Splitting original recordings before augmentation is recommended.

## Conclusion

This project demonstrates a deep learning-based approach to speech emotion recognition by integrating four speech datasets and combining acoustic feature extraction with a hybrid CNN-BiLSTM architecture.

The current training run achieved **78.81% reported evaluation accuracy**, demonstrating the potential of combining convolutional and recurrent neural networks for multi-class speech emotion classification.
