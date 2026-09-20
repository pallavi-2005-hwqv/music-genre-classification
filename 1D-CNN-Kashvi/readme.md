Absolutely — here’s a **README-ready methodology** based on exactly what you implemented with the MFCC 1D CNN.

````markdown
## Methodology – 1D CNN for Music Genre Classification

### 1. Dataset

The GTZAN Music Genre Dataset was used for music genre classification. The dataset contains 1,000 audio tracks belonging to 10 genres:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

Each track is approximately 30 seconds long and sampled at 22,050 Hz.

The dataset contains approximately 10,000 three-second audio segments. One corrupted audio file (`jazz.00054.wav`) was skipped during MFCC extraction, resulting in 9,980 usable segments from 999 original songs.

---

### 2. Data Preprocessing

The original audio files were processed individually. Each 30-second audio track was divided into 3-second segments.

For every segment:

1. The audio was loaded using Librosa.
2. The audio was sampled at 22,050 Hz.
3. 40 Mel-Frequency Cepstral Coefficients (MFCCs) were extracted.
4. A hop length of 512 was used.
5. Each segment produced an MFCC representation of shape `(130, 40)`.

Therefore, each audio segment was represented as a sequence of 130 time frames, with 40 MFCC features per frame.

The final MFCC dataset had the shape:

```text
(9980, 130, 40)
````

where:

* 9,980 = audio segments
* 130 = time frames
* 40 = MFCC features

---

### 3. Train, Validation and Test Split

To prevent data leakage, the original songs were divided into training, validation and testing sets before their corresponding segments were used.

The split consisted of:

* 70% training songs
* 15% validation songs
* 15% testing songs

This resulted in:

```text
Training:   6980 segments
Validation: 1500 segments
Testing:    1500 segments
```

The split was performed at the song level so that segments originating from the same song did not appear in different subsets.

---

### 4. Feature Scaling

The MFCC features were standardized using `StandardScaler`.

The scaler was fitted only on the training data and then applied to the validation and test data.

The input dimensions remained:

```text
Training:   (6980, 130, 40)
Validation: (1500, 130, 40)
Testing:    (1500, 130, 40)
```

Genre labels were converted from text labels into numerical class labels using `LabelEncoder`.

---

### 5. 1D CNN Architecture

A one-dimensional Convolutional Neural Network (1D CNN) was implemented using TensorFlow/Keras.

The model processes the MFCC representation as a temporal sequence and learns local patterns across consecutive time frames.

The architecture consists of:

1. **Conv1D**

   * 64 filters
   * Kernel size: 3
   * ReLU activation

2. **Batch Normalization**

3. **MaxPooling1D**

   * Pool size: 2

4. **Dropout**

   * Rate: 0.25

5. **Conv1D**

   * 128 filters
   * Kernel size: 3
   * ReLU activation

6. **Batch Normalization**

7. **MaxPooling1D**

   * Pool size: 2

8. **Dropout**

   * Rate: 0.25

9. **Conv1D**

   * 256 filters
   * Kernel size: 3
   * ReLU activation

10. **Batch Normalization**

11. **MaxPooling1D**

    * Pool size: 2

12. **Global Average Pooling 1D**

13. **Dense Layer**

    * 128 neurons
    * ReLU activation

14. **Dropout**

    * Rate: 0.5

15. **Output Layer**

    * 10 neurons
    * Softmax activation

The final model contained approximately **188,490 parameters**, of which **187,594 were trainable**.

---

### 6. Model Training

The CNN was trained using:

* Optimizer: Adam
* Loss function: Sparse Categorical Crossentropy
* Batch size: 32
* Maximum epochs: 30
* Early stopping patience: 5 epochs
* Best model weights restored using `restore_best_weights=True`

Early stopping was used to stop training when the validation loss stopped improving, helping reduce overfitting.

The model stopped training after 11 epochs.

---

### 7. Model Evaluation

The trained model was evaluated on the unseen test set using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The preliminary test results were:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 65.27% |
| Precision | 68.27% |
| Recall    | 65.27% |
| F1-score  | 65.51% |

A confusion matrix was generated to analyse classification performance across the 10 music genres.

Training and validation accuracy/loss curves were also generated to analyse the model's learning behaviour and potential overfitting.

---

### 8. Implementation Tools

The implementation was carried out using:

* Python
* TensorFlow / Keras
* Librosa 0.11.0
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Kaggle Notebook

```

**One small thing:** for your actual report, I'd call this **“MFCC-based 1D CNN”** throughout. That's more accurate than simply saying “1D CNN,” because MFCC extraction is a major part of your methodology.
```
