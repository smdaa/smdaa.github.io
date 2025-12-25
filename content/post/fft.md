+++
title = "TODO"
date = 2025-11-22
tags = ["todo"]
+++

{{< toc >}}

## Introduction

## The core idea

The Fourier transform correlates the signal with complex exponentials at different frequencies:

$$
X(f) = \int_{-\infty}^{\infty} x(t) e^{-i 2\pi f t} dt
$$

For each frequency $f$, we multiply the signal by $e^{-i 2\pi f t}$ and integrate. If that frequency is present, the product accumulates constructively. If not, oscillations cancel out.

$e^{-i 2\pi f t} = \cos(2\pi f t) - i \sin(2\pi f t)$ tests against sine and cosine simultaneously. A pure sine at frequency $f$ correlates zero with a cosine at $f$ (they're $90°$ out of phase), so we need both components.

The output $X(f)$ is complex: magnitude $|X(f)|$ is the amplitude at frequency $f$, phase $\angle X(f)$ is the timing offset.

## The discrete version

When signals are sampled at rate $f_s$, the Discrete Fourier Transform (DFT) replaces the continuous integral with a discrete sum over $N$ samples:

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi kn/N}
$$

This produces $N$ frequency coefficients at frequencies:

$$
f_k = \frac{k f_s}{N}, \quad k = 0,1,\dots,N-1
$$

The DFT implicitly treats the signal as periodic with period $N$ samples. In other words, the sequence $x[n]$ is assumed to repeat indefinitely in time. As a consequence, discontinuities at the boundaries of the $N$-sample segment can introduce spectral leakage, and the frequency resolution is limited to $\Delta f = f_s / N$. More details on these effects are discussed later in the practical considerations.

## Matrix form and change of basis

The DFT is a matrix multiplication $X = Wx$ where:

$$
W[k, n] = e^{-i 2\pi kn/N}
$$

We can view the DFT as a change of basis, expressing the signal in the coordinates of the complex exponential basis.

In fact the columns of the matrix $W$ are orthogonal:

$$
\sum_{n=0}^{N-1} e^{i 2\pi k_1 n/N} \cdot e^{-i 2\pi k_2 n/N} = \sum_{n=0}^{N-1} e^{i 2\pi (k_1-k_2)n/N}
$$

When $k_1 = k_2$, every term equals 1, giving $N$. When $k_1 \neq k_2$, this is a geometric series summing to zero (because $e^{i 2\pi(k_1-k_2)} = 1$).

Therefore:

$$
W^* W = NI
$$

With this we can infer that the inverse DFT is computed with the matrix
$$
W^{-1} = \frac{1}{N} W^*
$$

We can also infer the energy preservation, because:

$$
\|x\|^2 = x^* x
= \frac{1}{N} x^* (W^* W) x
= \frac{1}{N} (W x)^* (W x)
= \frac{1}{N} \|X\|^2
$$

In other words, representing a signal in the frequency domain preserves its total energy, up to a factor of $1/N$. This result is known as Parseval's theorem.


## Why complex exponentials?

We've seen that the DFT projects signals onto a basis of complex exponentials. But why these particular functions? After all, many orthogonal bases exist: wavelets, polynomials, or other function families could mathematically represent signals just as well.

Complex exponentials are special for two reasons: one physical, one algebraic.

Physically, many natural phenomena are inherently oscillatory. Sound waves, electromagnetic radiation, and mechanical vibrations are all naturally described by sinusoids. When we decompose a signal into frequency components, we're often uncovering the actual physical processes that generated it.

Another reason is algebraic: complex exponentials are the eigenfunctions of Linear Time Invariant (LTI) systems. This property makes them uniquely powerful for analyzing how signals transform through physical systems.

### LTI systems and convolution

An LTI system satisfies two properties:

1. Linearity: If the system produces output $y_1(t)$ for input $x_1(t)$ and output $y_2(t)$ for input $x_2(t)$, then for any scalars $a$ and $b$, the input $a x_1(t) + b x_2(t)$ produces output $a y_1(t) + b y_2(t)$.
2. Time invariance: If the system produces output $y(t)$ for input $x(t)$, then for any delay $\tau$, the input $x(t - \tau)$ produces output $y(t - \tau)$. The system's behavior does not change over time.

LTI systems appear throughout engineering and science: electrical circuits, acoustic spaces, optical systems, communication channels. Even when a system isn't perfectly linear or time-invariant, the LTI approximation often provides useful insights.

An LTI system is completely characterized by its impulse response $h(t)$ which is the output when the input is a unit impulse $\delta(t)$. In fact any input can be written as a weighted sum of shifted impulses:

$$
x(t) = \int_{-\infty}^{\infty} x(\tau) \delta(t - \tau) d\tau
$$

Using linearity and time invariance, the output becomes:

$$
y(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau
$$

In discrete time with circular boundary conditions, this becomes:

$$
y[m] = \sum_{n=0}^{N-1} x[n] h[(m-n) \bmod N]
$$

This circular convolution can be written as matrix multiplication $y = Hx$ where $H$ is a circulant matrix:

$$
H[m, n] = h[(m-n) \bmod N]
$$

$$
H = \begin{bmatrix}
h[0]   & h[N-1] & \cdots & h[1] \newline
h[1]   & h[0]   & \cdots & h[2] \newline
\vdots & \vdots & \ddots & \vdots \newline
h[N-1] & h[N-2] & \cdots & h[0]
\end{bmatrix}
$$

### Diagonalization by the DFT

The key property is that circulant matrices are diagonalized by the DFT matrix. To see why, we can compute the $(m, k)$ entry of the product $HW$:

$$
\begin{aligned}
(HW)[m, k]
&= \sum_{n=0}^{N-1} h[(m-n) \bmod N] e^{-i 2\pi nk/N}
\end{aligned}
$$

Let $r = (m - n) \bmod N$. Then $n \equiv (m - r) \pmod N$. Substituting:

$$
\begin{aligned}
(HW)[m, k] &= \sum_{r=0}^{N-1} h[r] e^{-i 2\pi (m-r)k/N} \\
&= e^{-i 2\pi mk/N}
\underbrace{\sum_{r=0}^{N-1} h[r] e^{i 2\pi rk/N}}_{\lambda_k}
\end{aligned}
$$

This shows $HW = W\Lambda$, where
$$
\Lambda = \text{diag}(\lambda_0, \dots, \lambda_{N-1}).
$$

Multiplying by $W^{-1}$ gives:

$$
W^{-1} H W = \Lambda
$$

The columns of $W$ are eigenvectors of $H$, and the eigenvalues $\lambda_k$ are simply the complex conjugate DFT of the impulse response.

Geometrically, a matrix multiplication typically rotates and scales a vector in complicated ways. Diagonalization finds a special coordinate system where the transformation only scales along each axis, with no rotation or mixing between dimensions.

For LTI systems, this eigenbasis consists of the complex exponentials. Passing a sinusoid through an LTI system cannot change its frequency; it only scales its amplitude and shifts its phase. The system's effect on each frequency component is independent and determined by a single complex number $\lambda_k$.

This is why frequency-domain analysis is so powerful: complicated time-domain convolution becomes simple multiplication in the frequency domain.

There is also a computational payoff. Diagonalization transforms the convolution $y = Hx$ from a dense matrix-vector product into element-wise multiplication. In the time domain, computing the convolution directly requires $O(N^2)$ scalar multiplications. Using the DFT's eigenproperty, we decompose the operation into three steps:

1. Transform input to frequency domain
2. Multiply by frequency response
3. Transform back to time domain

With the Fast Fourier Transform (FFT), steps 1 and 3 each take $O(N \log N)$ operations. The total complexity becomes $O(N \log N)$ instead of $O(N^2)$, a dramatic speedup for large $N$.

## The FFT algorithm

The Fast Fourier Transform (FFT) exploits symmetries in the DFT's complex exponentials to reduce computational complexity from $O(N^2)$ to $O(N \log N)$. 

It's not an approximation; it is an exact factorization of the DFT matrix $W_N$ into a product of sparse matrices.

To see this let's start with the definition of the DFT for size $N$ (assuming $N$ is even):

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi kn/N}
$$

We separate the summation index $n$ into even indices ($n=2m$) and odd indices ($n=2m+1$):

$$
\begin{aligned}
X[k] &= \sum_{m=0}^{N/2-1} x[2m] e^{-i 2\pi (2m)k/N} + \sum_{m=0}^{N/2-1} x[2m+1] e^{-i 2\pi (2m+1)k/N} \newline
&= \sum_{m=0}^{N/2-1} x[2m] e^{-i 2\pi mk/(N/2)} + e^{-i 2\pi k/N} \sum_{m=0}^{N/2-1} x[2m+1] e^{-i 2\pi mk/(N/2)}
\end{aligned}
$$

Notice that these two summations are themselves length-$N/2$ DFTs of the even indexed and odd indexed samples. 

Let $E[k]$ be the DFT of the even samples and $O[k]$ be the DFT of the odd samples. The expression simplifies to:

$$
X[k] = E[k] + \omega^k O[k]
$$

Where $\omega = e^{-i 2\pi / N}$ is the "twiddle factor."

Due to the periodicity of the DFT, $E[k + N/2] = E[k]$ and $O[k + N/2] = O[k]$. However, the twiddle factor undergoes a phase shift: $\omega^{k + N/2} = e^{-i 2\pi (k+N/2)/N} = e^{-i 2\pi k/N} e^{-i \pi} = -\omega^k$. This symmetry allows us to compute two values of $X$ for the cost of one:

$$
\begin{aligned}
X[k] &= E[k] + \omega^k O[k] \newline
X[k + N/2] &= E[k] - \omega^k O[k]
\end{aligned}
$$

