# LSTM Music Genre Classification

This repository contains the LSTM implementation for a 4-model comparative deep learning project evaluating MLP, 1D CNN, LSTM, and Transformer architectures on music genre classification. The LSTM model processes sequential Mel-Frequency Cepstral Coefficients (MFCCs) to capture temporal dependencies across audio frames.

## Dataset
The project uses the GTZAN Music Genre Dataset containing 1,000 30-second audio tracks across 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. One corrupted track (`jazz.00054.wav`) is safely skipped.

## Preprocessing
- Audio is split into 3-second non-overlapping segments sampled at 22,050 Hz.
- 40 MFCC features are extracted with `n_fft = 2048` and `hop_length = 512`, producing an input shape of `(130, 40)` per segment.
- A song-level stratified split (70% train, 15% validation, 15% test) ensures no segment leakage across subsets.
- Features are scaled using `StandardScaler` fitted strictly on the training set.

## Model
The model is a 2-layer LSTM network:
- Input: `(130, 40)`
- LSTM Layer 1: 128 units (`return_sequences=True`) + Dropout(0.3)
- LSTM Layer 2: 64 units (`return_sequences=False`) + Dropout(0.3)
- Dense Layer: 64 units (ReLU) + Dropout(0.3)
- Output Layer: 10 units (Softmax)
- Total Parameters: 140,746 (all trainable)

## Training
- Optimizer: Adam (learning rate = 0.001)
- Loss: Sparse Categorical Crossentropy
- Batch Size: 32
- Maximum Epochs: 30
- Early Stopping: `val_loss` monitor with `patience = 5` and `restore_best_weights = True` (stopped at epoch 15, best weights from epoch 10)

## Results

| Metric | Score |
|---|---|
| Accuracy | 59.20% |
| Precision (Weighted) | 58.61% |
| Recall (Weighted) | 59.20% |
| F1-score (Weighted) | 58.70% |

## Files
- `lstm_genre_classification.ipynb`: Complete self-contained Jupyter notebook containing all pipeline stages, model definition, training execution, evaluation, and visualizations.
- `LSTM_METHODOLOGY.md`: Detailed methodology and literature notes for the LSTM model.
- `results/lstm_accuracy.png`: Training vs. validation accuracy curve.
- `results/lstm_loss.png`: Training vs. validation loss curve.
- `results/lstm_confusion_matrix.png`: Heatmap confusion matrix on the unseen test set.

## References
1. Suman Kumar Swarnkar and Yogesh Kumar Rathore (2024), *"Music Genre Classification Using Long Short-Term Memory (LSTM) Networks: Analyzing Audio Spectrograms for Enhanced Multimedia Understanding,"* in *Machine Learning in Multimedia*, CRC Press.
2. Nantalira Niar Wijaya, De Rosal Ignatius Moses Setiadi, and Ahmad Rofiqul Muslikh (2024), *"Music-Genre Classification using Bidirectional Long Short-Term Memory and Mel-Frequency Cepstral Coefficients,"* *Journal of Computing Theories and Applications (JCTA)*, Vol. 1, No. 3, pp. 263-274.
