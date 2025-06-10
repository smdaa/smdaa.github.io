---
title: Building a Keystroke Audio Classifier
pubDatetime: 2025-06-10
description: Building a Keystroke Audio Classifier
featured: true
tags:
  - machine-learning
  - audio-processing
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