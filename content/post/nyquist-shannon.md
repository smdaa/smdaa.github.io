+++
title = "Why you need to sample at twice the frequency (and when you don't)"
date = 2025-11-20
tags = ["todo"]
+++

{{< toc >}}

# Introduction 

The Nyquist–Shannon theorem is often the first concept introduced in signal processing. It states that if a signal has no frequencies above $f$, then sampling at $2f$ is enough to determine it completely. In other words, to reconstruct a signal without losing information, you must sample at least twice its highest frequency.  

In this post, we will look at an intuition behind this theorem by focusing on what happens in the frequency domain. 

We will also look at cases where the Nyquist–Shannon criteria can be beaten.

The animations below were by made with the library [Manim](https://www.manim.community/) and can be found [here](https://github.com/smdaa/dsp-manifesto/tree/main/nyquist-shannon).

# Aliasing  

Aliasing is what happens when we sample too slowly. When the sampling rate falls below the Nyquist rate ($2f$), the sampled points can no longer distinguish the original signal from lower-frequency imposters. 

In the animation below, we start with a 3 Hz signal and gradually reduce the sampling rate. At $5Hz$ below the Nyquist rate of $6Hz$ the samples now trace out a 2 Hz wave instead. The original 3 Hz signal has been "aliased" to a completely different frequency. 

{{< video src="/assets/nyquist-shannon/aliasing_animation.mp4" type="video/mp4" >}}

# From Time-Domain Sampling to Frequency-Domain Replicas

Sampling a continuous signal is mathematically equivalent to multiplying it by a sampling function. This sampling function is a train of Dirac delta functions spaced at intervals of $T_s = \frac{1}{f_s}$:
$$s(t) = \sum_{n=-\infty}^{\infty} \delta(t - nT_s)$$
where $f_s$ is the sampling rate.


The fourier transform of the sampling function is:
$$
S(f) = \int_{-\infty}^{\infty} \left(\sum_{n=-\infty}^{\infty} \delta(t - nT_s)\right) e^{-i 2\pi f t}\, dt
$$
$$
S(f) = \sum_{n=-\infty}^{\infty} \int_{-\infty}^{\infty} \delta(t - nT_s)\, e^{-i 2\pi f t}\, dt
$$
Using the shifting property of the dirac:
$$
\int_{-\infty}^{\infty} \delta(t - nT_s)\, e^{-i 2\pi f t}\ dt = e^{-i 2\pi f (nT_s)}
$$
We have
$$
S(f) = \sum_{n=-\infty}^{\infty} e^{-i 2\pi f n T_s}
$$
Using the Poisson summation identity:
$$
\sum_{n=-\infty}^{\infty} e^{-i 2\pi f n T_s}
= \frac{1}{T_s} \sum_{k=-\infty}^{\infty} \delta\!\left(f - k f_s\right)
$$
Therefore:
$$
S(f) = \frac{1}{T_s} \sum_{k=-\infty}^{\infty} \delta\!\left(f - k f_s\right)
$$
The result indicates that sampling a signal in the time domain leads to periodic repetitions of its spectrum in the frequency domain, since multiplying in time corresponds to convolution in frequency.

In other words, sampling creates copies of the original spectrum at multiples of $f_s$. When these copies overlap, aliasing occurs. The animation below shows this: as $f_s$ decreases, the replicas move closer until they collide at $f_s < 2 \times bandwidth$.

{{< video src="/assets/nyquist-shannon/sampling_frequency_visualisation.mp4" type="video/mp4" >}}

# Compressive sensing


