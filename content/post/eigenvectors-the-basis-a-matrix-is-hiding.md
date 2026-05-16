+++
title = "Eigenvectors: the basis a matrix is hiding"
date = 2026-05-13
tags = ["fourier-transform", "signal-processing", "linear-algebra", "fft"]

+++

{{< toc >}}

## Introduction

Eigenvectors are defined with the following equation:

$$
Av = \lambda v
$$

A vector $v$ is an eigenvector of matrix $A$ if applying $A$ leaves its direction unchanged and only scales it by $\lambda$.

Another way to look at it: collect the eigenvectors of $A$ into a basis and in that basis, $A$ is diagonal. It does not rotate, shear, or mix. It only stretches each direction independently. The complexity you see in standard coordinates is not intrinsic to $A$, it is a consequence of looking at it in the wrong basis.

This is why eigenvectors appear everywhere. PageRank, facial recognition, structural mechanics, they are all asking the same question: what are the natural coordinates of this system? The coordinates in which it becomes simple? The answer is always the eigenvectors.

In this article we make this concrete through three examples. First Fibonacci, where iteration visibly converges to the natural coordinates. Then a physical system of coupled oscillators, where the eigenbasis separates motions that would otherwise mix. Then PCA, where it decorrelates data. Three domains, one idea.

## Example 1: Fibonacci and why the golden ratio always wins

Starting from $0$ and $1$, each term in the Fibonacci sequence is the sum of the two before it: $0,\ 1,\ 1,\ 2,\ 3,\ 5,\ 8,\ 13,\ 21, \dots$

The Fibonacci recurrence $F_{n+1} = F_n + F_{n-1}$ can be written as a matrix iteration:

$$
\begin{pmatrix} F_{n+1} \newline F_n \end{pmatrix}
= \begin{pmatrix} 1 & 1 \newline 1 & 0 \end{pmatrix}
\begin{pmatrix} F_n \newline F_{n-1} \end{pmatrix}
$$

We will call this $v_{n+1} = Av_n$.

In the animation below, we apply $A$ repeatedly and plot each vector. Two things happen: the vector grows, and it rotates toward a fixed line. Once the direction has converged, each new vector is just $\varphi$ times the previous one.

{{< video src="/assets/eigenvectors-the-basis-a-matrix-is-hiding/fibonacci_eigenvector.webm" type="video/webm" >}}

The starting pair does not matter. Try $(17, 50)$: the same line.

### Why it works and what it gives you

Solving $Av = \lambda v$ gives two eigenvalues, $\varphi = \frac{1+\sqrt{5}}{2}$ and $\psi = \frac{1-\sqrt{5}}{2}$, with eigenvectors $v_1 = \begin{pmatrix} \varphi \newline 1 \end{pmatrix}$ and $v_2 = \begin{pmatrix} \psi \newline 1 \end{pmatrix}$. Because $v_1$ and $v_2$ point in different directions they form a basis, which means we can write 

$$A = P D P^{-1}$$ 

with $D$ diagonal. 

Raising both sides to the $k$-th power then turns matrix multiplication into scalar exponentiation:

$$
A^k = P \begin{pmatrix} \varphi^k & 0 \newline 0 & \psi^k \end{pmatrix} P^{-1}
$$

Working this out for the Fibonacci matrix yields an exact closed form for the $n$-th term:

$$
F_n = \frac{\varphi^n - \psi^n}{\sqrt{5}}
$$

This is Binet's formula. No iteration, no recurrence. It also explains the animation: since $|\psi| < |\varphi|$, the $\psi$ term vanishes and the direction locks onto the eigenvector of $\varphi$. The convergence you see is a direct consequence of one eigenvalue being larger than the other.

Fibonacci is one instance of a general pattern. Whenever you apply a linear map repeatedly, the long-run behavior is governed by the eigenvalues: the largest one grows fastest, and its direction eventually absorbs everything else. PageRank works this way. So do population models, Markov chains, and any system defined by iterating a rule. The eigenbasis is what makes that behavior readable.

## Example 2: Coupled oscillators and the natural coordinates of motion

## Example 3: PCA and the natural coordinates of data

## Conclusion

