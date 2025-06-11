---
title: Building a Keystroke Audio Classifier
pubDatetime: 2025-06-10
description: Building a Keystroke Audio Classifier
featured: true
tags:
  - machine-learning
  - audio-processing
  - cpp
  - python
---

## Table of contents

## Introduction

This short project explores machine learning methods for classifying keystrokes based on audio. The goal is to determine which key the user is pressing by only analyzing the sound it produces. First, we will look at the data collection method that captures keyboard sounds with precise timing. Then, we will examine different ML algorithms: MFCC features with traditional classifiers, 1D CNN on raw audio, and neural networks on FFT spectrograms.

## Data Collection
The keystroke audio collection tool is built with a dual-threaded architecture for precision and responsiveness.

* Audio Thread: Continuously captures microphone input using PortAudio at 44.1 kHz. The audio is stored in a circular buffer that maintains a rolling window of the last 5 seconds of audio data.

* Keyboard Thread: Monitors key presses in real time using the Windows API. When a key is detected, the system notes the timestamp.

With the timestamped key press event, a ~400 ms window of audio (100 ms before and 300 ms after the keypress) is extracted from the buffer and saved. Each sample is labeled with the corresponding key.
This architecture ensures that every keystroke is paired with a clean, time-aligned audio snippet.

You can find the full source code for the keystroke audio capture tool [here](https://github.com/smdaa/keystroke-audio-classifier/tree/main/KeySoundCapture)

## A Look At The Raw Data

## MFCC Feature Extraction with Traditional Classifiers
The first approach leverages Mel-Frequency Cepstral Coefficients (MFCC) as feature vectors, implementing the complete mathematical pipeline from scratch:

**Signal Windowing:** The audio signal is segmented using overlapping Hamming windows of 25ms duration with 10ms hop size.

**Spectral Analysis:** The windowed signals undergo FFT transformation with 1024 points, producing the power spectrum:

$$P(k) = |X(k)|^2$$

where $X(k)$ is the FFT of the windowed signal.

**Mel-Scale Transformation:** Linear frequencies are converted to the perceptually-motivated mel-scale using:

$$m = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)$$

The inverse transformation is:

$$f = 700 \times \left(10^{m/2595} - 1\right)$$

**Mel Filterbank:** A bank of 26 triangular filters is constructed with mel-spaced center frequencies. Each filter $H_m(k)$ has the form:

$$
H_m(k) = \begin{cases}
0 & \text{if } k < f(m-1) \\
\frac{k - f(m-1)}{f(m) - f(m-1)} & \text{if } f(m-1) \leq k < f(m) \\
\frac{f(m+1) - k}{f(m+1) - f(m)} & \text{if } f(m) \leq k < f(m+1) \\
0 & \text{if } k \geq f(m+1)
\end{cases}
$$

![](/assets/building-a-keystroke-audio-classifier/mel-filterbank.png)

Mel filter banks mimic human hearing by allocating more filters to lower frequencies where we're more sensitive, and fewer filters to higher frequencies where our discrimination is coarser. Also instead of using all 512+ FFT bins, mel filter banks compress the spectrum into 26 perceptually-relevant channels

**Mel Energy Computation:** The mel-filtered energies are calculated as:

$$S(m) = \sum_{k=0}^{N/2} P(k) \cdot H_m(k)$$

**Logarithmic Compression:** Dynamic range compression is applied:

$$\log S(m) = \log(S(m))$$

**Discrete Cosine Transform:** The final MFCC coefficients are obtained through Type-II DCT:

$$C(n) = \sqrt{\frac{2}{M}} \sum_{m=0}^{M-1} \log S(m) \cos\left(\frac{\pi n (m + 0.5)}{M}\right)$$

where $M = 26$ is the number of mel filters and $n = 0, 1, ..., 12$ for the first 13 coefficients.

**Feature Aggregation:** For each keystroke sample, the mean MFCC vector across all time frames serves as the final feature representation:

$$\bar{C}(n) = \frac{1}{T} \sum_{t=0}^{T-1} C_t(n)$$

where $T$ is the number of time frames and $C_t(n)$ is the $n$-th MFCC coefficient at time frame $t$.

These 13-dimensional feature vectors are then fed to traditional classifiers including K-Nearest Neighbors, SVM, Decision Tree, and Random Forest, evaluated through 5-fold cross-validation with randomized hyperparameter search to establish a baseline for keystroke classification performance.

The following table summarizes the cross-validation performance of different classifiers on MFCC features:

| Model         | Train Accuracy  | Test Accuracy   |
| ------------- | --------------- | --------------- |
| KNN           | 1.0000 ± 0.0000 | 0.8084 ± 0.0051 |
| SVM           | 0.9510 ± 0.0032 | 0.8588 ± 0.0103 |
| Decision Tree | 0.6098 ± 0.0082 | 0.5255 ± 0.0080 |
| Random Forest | 0.9980 ± 0.0001 | 0.7913 ± 0.0097 |

SVM achieves the best test accuracy (85.88%) with reasonable training accuracy, indicating good generalization