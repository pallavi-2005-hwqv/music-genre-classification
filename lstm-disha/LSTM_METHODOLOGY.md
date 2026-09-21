# LSTM Methodology

### 1. Dataset and Preprocessing
We use the GTZAN dataset consisting of 1,000 audio tracks (approximately 30 seconds each) across 10 musical genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. One corrupted audio file (`jazz.00054.wav`) is skipped safely during loading, leaving 999 usable songs.

To provide sufficient sequential data for the recurrent network and align with the other models in the project, each 30-second audio track is split into 3-second non-overlapping segments. Each segment is sampled at a sampling rate of 22,050 Hz.

### 2. MFCC Representation
For each 3-second segment, we extract 40 Mel-Frequency Cepstral Coefficients (MFCCs) using an FFT window size of 2,048 samples and a hop length of 512 samples. 

This transformation converts each 3-second audio slice into a 2D time-frequency matrix of shape (40, 130). We transpose this matrix to (130, 40) so that the first dimension represents 130 sequential time frames and the second dimension contains the 40 MFCC features per frame. The full dataset is represented as a 3D tensor of shape `(samples, 130, 40)`.

### 3. Data Splitting
To strictly avoid data leakage across segments, splitting is performed at the song level before segmentation:
- **70% Training**: 700 songs (6,980 segments)
- **15% Validation**: 150 songs (1,500 segments)
- **15% Testing**: 150 songs (1,500 segments)

The split is stratified by genre class and uses a fixed random seed (`random_state=42`). Segments from any given song belong exclusively to a single split.

### 4. Feature Scaling
Feature scaling is performed using `StandardScaler`. The scaler is fitted strictly on the training segment feature vectors and then used to transform the validation and test sets. 

The 3D tensor `(samples, 130, 40)` is flattened to 2D `(samples * 130, 40)`, standardized, and reshaped back to `(samples, 130, 40)` so that feature scales are standardized without altering temporal sequence dimensions.

### 5. LSTM Architecture
The model is implemented in TensorFlow/Keras as a sequential recurrent neural network:
- **Input Layer**: Shape `(130, 40)`
- **LSTM Layer 1**: 128 units, `return_sequences=True` (captures lower-level temporal transitions across frames)
- **Dropout Layer 1**: Rate 0.3 (regularization)
- **LSTM Layer 2**: 64 units, `return_sequences=False` (aggregates sequence representation into a summary vector)
- **Dropout Layer 2**: Rate 0.3
- **Dense Layer 1**: 64 units with ReLU activation
- **Dropout Layer 3**: Rate 0.3
- **Output Dense Layer**: 10 units with Softmax activation

**Model Parameters**:
- Total parameters: 140,746
- Trainable parameters: 140,746
- Non-trainable parameters: 0

### 6. Training Configuration
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Maximum Epochs**: 30
- **Early Stopping**: Monitored on `val_loss` with `patience=5` and `restore_best_weights=True`. Training stopped at epoch 15, restoring the best model weights from epoch 10.

### 7. Evaluation Metrics
The model is evaluated on the unseen test set (1,500 segments) using:
- **Accuracy**: 59.20% (0.5920)
- **Precision (weighted)**: 58.61% (0.5861)
- **Recall (weighted)**: 59.20% (0.5920)
- **F1-score (weighted)**: 58.70% (0.5870)

### 8. Brief Error Analysis
From the test confusion matrix:
- **Best Performing Genres**: Classical (143/150 correct, 95.3%) and Metal (118/150 correct, 78.7%) have the highest classification rates due to distinct acoustic timbre and distortion patterns. Pop (103/150) and Jazz (100/150) also perform reliably.
- **Main Confusions**:
  - Rock is the most confused genre (42/150 correct, 28.0%), frequently misclassified as Country (29 segments) and Disco (24 segments) due to overlapping guitar instrumentation and song structures.
  - Disco is frequently confused with Pop (29 segments) and Rock (25 segments) due to shared tempo and rhythmic patterns.
  - Hiphop and Reggae show mutual confusion (19 Hiphop segments predicted as Reggae, 15 Reggae segments predicted as Hiphop) due to similar basslines and syncopated beats.

---

### Literature Notes

#### Paper 1
- **Citation**: Suman Kumar Swarnkar and Yogesh Kumar Rathore (2024), *"Music Genre Classification Using Long Short-Term Memory (LSTM) Networks: Analyzing Audio Spectrograms for Enhanced Multimedia Understanding,"* in *Machine Learning in Multimedia*, CRC Press.
- **Problem**: Classifying music genres from audio signals by capturing sequential temporal patterns across spectral representations.
- **Dataset**: GTZAN dataset (10 genre classes).
- **LSTM Method**: Evaluates LSTM networks applied to sequential audio feature representations extracted from audio spectrograms.
- **Main Result**: Demonstrates that recurrent architectures can effectively model temporal dependencies in musical structures compared to static feature classifiers.
- **Relevance**: Supports our project's approach of treating audio frame features sequentially rather than flattening them into static summaries.

#### Paper 2
- **Citation**: Nantalira Niar Wijaya, De Rosal Ignatius Moses Setiadi, and Ahmad Rofiqul Muslikh (2024), *"Music-Genre Classification using Bidirectional Long Short-Term Memory and Mel-Frequency Cepstral Coefficients,"* *Journal of Computing Theories and Applications (JCTA)*, Vol. 1, No. 3, pp. 263-274.
- **Problem**: Improving classification accuracy for audio genres by combining cepstral features with recurrent models.
- **Dataset**: GTZAN dataset (1,000 audio tracks, 10 genres).
- **LSTM Method**: Extracts MFCC features across audio segments and applies recurrent LSTM architectures to capture temporal transitions.
- **Main Result**: Shows that MFCC sequences paired with recurrent layers capture both spectral envelope characteristics and temporal flow across music tracks.
- **Relevance**: Validates our choice of 40-MFCC feature sequences as the input representation for the standard LSTM model.
