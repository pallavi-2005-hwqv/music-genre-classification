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
