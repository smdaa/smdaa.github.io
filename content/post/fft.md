+++
title = "TODO"
date = 2025-11-22
tags = ["todo"]
+++

{{< toc >}}

## Introduction

A fundamental idea in signal processing is that signals can be represented using oscillations.

Periodic signals can be written as sums of sinusoids via Fourier series, while general nonperiodic signals can be described as an continuous superposition of complex exponentials via the Fourier transform.

In practice (where we work with sampled, finite-length data) this becomes the discrete Fourier transform (DFT), which expresses any finite sequence (audio, image, or any timeseries data) as a weighted sum of discrete frequencies.

Seen through a linear-algebra lens, this is simply a change of basis.

TODO: write hook

## The math behind the fourier transform

The fourier transfom for a signal $x$ is defined as follows:

$$
X(f) = \int_{-\infty}^{\infty} x(t)\ e^{-j 2\pi f t}\ dt
$$

At each frequency $f$, we multiply the signal $x(t)$ by a complex exponential $e^{-j2\pi ft}$ and integrate the result. The output $X(f)$ tells us how much of that frequency is present in the signal.

The complex exponential is $e^{-j2\pi ft} = \cos(2\pi ft) - j\sin(2\pi ft)$, so we're actually multiplying with both cosine and sine simultaneously. This is essential: if we only used cosine, a signal at frequency $f$ but phase-shifted by $\pi/2$ would give zero and appear absent. The sine component catches it.

We can think of this as a correlation operation: if $x(t)$ contains frequency $f$, the product $x(t) \cdot e^{-j2\pi ft}$ accumulates constructively, giving a large $|X(f)|$. If $x(t)$ doesn't contain frequency $f$, the product oscillates between positive and negative and cancels to nearly zero. The magnitude $|X(f)|$ tells you how much of that frequency is present, while the phase $\angle X(f)$ tells you its timing offset.

## From continuous to discrete

In practice, the continuous signal $x(t)$ becomes a finite sequence of samples $x[n]$.

The Discrete Fourier Transform (DFT) is the sampled counterpart of the Fourier transform:
$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi \frac{k}{N} n}
$$

where $k = 0, 1, \ldots, N-1$ are the discrete frequency bins.

The integral becomes a sum, and instead of a continuous spectrum $X(f)$, we get $N$ discrete frequency coefficients $X[k]$. Each $k$ represents a frequency $f_k = \frac{k}{N} f_s$, where $f_s$ is the sampling rate.

## The DFT as a matrix

We can rewrite the DFT formula as a matrix-vector multiplication:

$$
X = W x
$$

where $x$ is the $N \times 1$ vector of time-domain samples, $X$ is the $N \times 1$ vector of frequency coefficients, and $W$ is the $N \times N$ DFT matrix with entries:

$$
W[k,n] = e^{-j 2\pi \frac{k}{N} n}
$$

This is a change of basis.

To see why $W$ forms a valid basis for $\mathbb{C}^N$, note that $W$ is made of $N$ orthogonal columns:

The dot product of any two distinct columns is:

$$
\langle W_{:,k_1}, W_{:,k_2} \rangle = \sum_{n=0}^{N-1} e^{j 2\pi \frac{k_1}{N} n} \cdot e^{-j 2\pi \frac{k_2}{N} n} = \sum_{n=0}^{N-1} e^{j 2\pi \frac{k_1 - k_2}{N} n}
$$

If $k_1 \neq k_2$, let $r = e^{j 2\pi \frac{k_1 - k_2}{N}}$ and denote the sum as $S$. Then:

$$
S = 1 + r + r^2 + \cdots + r^{N-1}
$$

Multiply both sides by $r$:

$$
rS = r + r^2 + r^3 + \cdots + r^N
$$

Subtract $S$:

$$
rS - S = r^N - 1
$$

Then:

$$
S = \frac{r^N - 1}{r - 1}
$$

Therefore:

$$
S = \frac{e^{j 2\pi (k_1 - k_2)} - 1}{e^{j 2\pi \frac{k_1 - k_2}{N}} - 1} = \frac{1 - 1}{e^{j 2\pi \frac{k_1 - k_2}{N}} - 1} = 0
$$

since $e^{j 2\pi (k_1 - k_2)} = 1$ for any integer difference (a full rotation around the unit circle).

When $k_1 = k_2$, the sum equals $N$ (all terms are $e^0 = 1$).

**Unitarity:** We've shown that:
$$
\langle W_{:,k_1}, W_{:,k_2} \rangle = \begin{cases}
N & \text{if } k_1 = k_2 \\
0 & \text{if } k_1 \neq k_2
\end{cases}
$$

The $(k_1, k_2)$ entry of the matrix product $W^* W$ is precisely $\langle W_{:,k_1}, W_{:,k_2} \rangle$. Therefore:
$$
W^* W = N I
$$




















## Properties of the DFT Matrix

- **Unitary:** W*W† = N·I (energy preserved, orthogonal basis)
  - Parseval's theorem: Σ|x[n]|² = (1/N)Σ|X[k]|²
  - Information is preserved, just represented differently
- **Symmetric:** lots of internal structure
  - Exploited by FFT algorithm
  - Conjugate symmetry for real signals
- **Circulant matrices are diagonalized by DFT**
  - Convolution matrix Circ(h) becomes: W† · Circ(h) · W = Diagonal
  - This is eigendecomposition: W† gives eigenvectors, diagonal gives eigenvalues
  - **Why convolution becomes multiplication:** changing to basis where operator is diagonal
  - Circulant matrices have sinusoids as eigenvectors
  - Eigenvalues are the DFT of first column

## The FFT Algorithm

- Naive DFT: O(N²) operations (matrix-vector multiply)
- Cooley-Tukey: exploit symmetry in W
  - Split even/odd samples
  - Recursively compute smaller DFTs
  - Combine with twiddle factors
- Complexity: O(N log N)
- *Butterfly diagram for N=8*
- One of the most important algorithms in computing
- Powers of 2 are optimal but not required
- Many variants: radix-4, split-radix, prime-factor algorithm
- Modern implementations highly optimized (FFTW)

## Practical Considerations

- **Real signals:** conjugate symmetry → only compute N/2 frequencies (RFFT)
  - X[N-k] = X[k]* for real x[n]
  - Save factor of 2 in computation and storage
- **Windowing:** rectangular window causes spectral leakage
  - Sharp edges in time → spread in frequency
  - Windows: Hann, Hamming, Blackman taper the edges
  - Trade-off: main lobe width vs side lobe level
- **Zero-padding:** interpolates frequency domain (doesn't add information)
  - Smooth frequency spectrum appearance
  - Does not improve frequency resolution
- **Nyquist frequency:** maximum detectable frequency = fs/2
  - Higher frequencies alias to lower ones
  - Anti-aliasing filter needed before sampling
- **Normalization:** conventions vary (1/N on forward, inverse, or both)
  - Check library documentation
  - Affects Parseval's theorem formula

## Common Transform Pairs

- Impulse ↔ Constant (delta-constant duality)
- Sinusoid ↔ Single peak
- Box/Rectangle ↔ Sinc
- Gaussian ↔ Gaussian (self-dual under FT)
- Narrow in time ↔ Wide in frequency (uncertainty principle)
- Derivative in time ↔ Multiplication by jω in frequency
- Convolution in time ↔ Multiplication in frequency
- Shift in time ↔ Phase shift in frequency

## Project: Poisson Image Editing

### What is Poisson's Equation?

- Poisson's equation: ∇²u = f
- ∇² is the Laplacian operator (measures how curved/smooth a function is)
- In 2D images: ∂²u/∂x² + ∂²u/∂y² = f
- u is the solution (output image), f is what we specify (source term)
- Fundamental equation in physics: electrostatics, fluid flow, heat diffusion
- In images: describes how pixel values relate to their neighbors

### Why Use It for Images?

- **Key insight:** Edit gradients (how fast pixels change), not pixels directly
- Human vision is sensitive to edges and contrast, not absolute brightness
- Preserving gradients = preserving texture and detail
- Allows blending objects with different brightness levels seamlessly
- Weber's law: we perceive relative changes, not absolute values
- Gradient domain gives more perceptual control

### The Seamless Cloning Problem

- Goal: paste region from source image S into target image T
- Direct copy-paste creates visible seams (brightness mismatch)
- **Poisson approach:**
  - Inside pasted region: preserve gradients from S (keep detail/texture)
  - At boundary: match pixel values of T (no seam)
  - Mathematically: solve ∇²u = ∇²S with boundary condition u|∂Ω = T|∂Ω
- Result: seamless blend that preserves both source detail and target context
- Used in professional editing software (Photoshop content-aware fill)

### Why This is Hard

- Each pixel gives one equation
- For N pixels, that's N equations with N unknowns
- Forms a sparse linear system: Au = b
- Direct solve: O(N²) to O(N³) - too slow for real-time
- Matrix A is sparse (each pixel only depends on neighbors)
- But even sparse solvers are slow for large images
- Iterative methods (Jacobi, Gauss-Seidel) converge slowly

### How FFT Makes It Fast

- **Laplacian is a convolution:** ∇²u = u ⊗ kernel where kernel = [0, 1, 0; 1, -4, 1; 0, 1, 0]
- Discrete Laplacian approximates second derivatives
- **In frequency domain:** convolution becomes multiplication
  - ℱ{∇²u} = ℱ{kernel} · ℱ{u}
- **Solving in frequency domain:**
  - ℱ{u} = ℱ{f} / ℱ{kernel}
  - u = ℱ⁻¹{ℱ{f} / ℱ{kernel}}
- **Complexity:** O(N log N) instead of O(N³)
- Each frequency component solves independently (diagonal in Fourier basis)
- Division by zero at DC (k=0): handle separately with boundary conditions
- Works because problem has special structure (translational invariance)

### Why This Connects to FFT Fundamentals

- Laplacian eigenfunctions are sinusoids (same as Fourier basis!)
- This is the **eigenfunction property** from earlier
- Differential operators become multiplication in frequency domain
- FFT transforms calculus (PDEs) into algebra (division)
- Each sinusoid is processed independently by Laplacian
- Eigenvalue of frequency k: -4(sin²(πk/N) in x and y directions)
- Diagonalization makes solving trivial

### Applications Beyond Cloning

- Object removal: fill holes by solving with surrounding boundary
- Illumination correction: flatten lighting, keep texture
- HDR tone mapping: compress dynamic range without losing detail
- Document scanning: remove lighting variations, flatten texture
- Gradient domain editing: manipulate gradients, reconstruct image
- Panorama stitching: blend overlapping images seamlessly
- Shadow removal: flatten shadows while preserving edges
- Image matting: extract foreground with natural boundaries

### The Demo

**User interaction:**

- Upload background image and object to paste
- Select region to clone (mask)
- Click position in background
- Instant seamless blend
- Toggle to compare: direct paste vs Poisson blend

**What happens behind the scenes:**

- Extract gradient field (∇S) from source region
- Compute Laplacian of source: ∇²S
- Set boundary values from target image
- FFT(∇²S) → divide by FFT(Laplacian kernel) → IFFT
- Handle DC component separately (preserve average intensity)
- Result: perfect blend in milliseconds
- Process each color channel independently

**Why surprising:**

- FFT solves calculus, not just "frequency analysis"
- Enables real-time PDE solving
- Same math as audio processing, completely different application
- Professional image editing software uses this (Photoshop content-aware fill)
- Bridge between signal processing and computational photography
- Shows power of thinking in frequency domain

**Implementation notes:**

- Use 2D FFT (separable: FFT rows then columns)
- Handle boundary conditions carefully
- Periodic boundary assumption can cause artifacts at edges
- May need to extend domain or use DCT instead
- RGB processed separately or convert to LAB color space

## Applications

- Audio: MP3 compression, pitch detection, effects (reverb, EQ)
- Images: JPEG (DCT variant), filtering, analysis
- Communication: OFDM (WiFi/LTE), channel equalization
- Science: spectroscopy, MRI, radio astronomy
- Math: fast polynomial multiplication, convolution, PDEs
- Computer graphics: texture synthesis, seamless tiling
- Medical imaging: CT reconstruction, MRI processing
- Geophysics: seismic data analysis

## Extensions

- **2D FFT:** images (separable: rows then columns)
- **STFT:** time-varying signals (spectrograms)
- **DCT:** real-valued, used in JPEG/MP3
- **Wavelets:** better time-frequency localization
- **Non-uniform FFT (NUFFT):** irregular sampling
- **Fractional Fourier transform:** rotation in time-frequency plane
- **Chirp Z-transform:** zoom into frequency region of interest

## Conclusion

- FFT = change of basis, from time to frequency
- Sinusoids are special: orthogonal, complete, eigenfunctions
- O(N log N) makes it practical
- Applications go far beyond spectrum analysis
- Enables real-time solving of problems that seem intractable
- Same mathematical framework across diverse domains
- Understanding basis perspective unlocks deeper insights
