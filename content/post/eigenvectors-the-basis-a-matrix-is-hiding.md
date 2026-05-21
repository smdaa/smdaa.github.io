+++
title = "Eigenvectors: the basis a matrix is hiding"
date = 2026-05-13
tags = ["linear-algebra", "signal-processing"]

+++

{{< toc >}}

## Introduction

Eigenvectors are defined as follows: a vector $v$ is an eigenvector of matrix $A$ if

$$Av = \lambda v$$

Meaning applying $A$ only scales $v$ by $\lambda$, it does not rotate it. 

Eigenvectors matter because when they form a basis, they give you the coordinates where $A$ becomes diagonal. In those coordinates, nothing gets mixed together. Each eigenvector is its own direction, and $A$ just stretches or shrinks along it.

If $A$ has $n$ linearly independent eigenvectors, we can put them into a matrix $P$ and write $A = PDP^{-1}$, where $D$ is diagonal. That means that, in the eigenvector basis, $A$ has no cross-talk between coordinates. Each axis changes independently.

TODO

## Example 1: The Fibonacci matrix

The Fibonacci sequence starts with $F_0 = 0$, $F_1 = 1$, and each subsequent term is the sum of the two before it: $0, 1, 1, 2, 3, 5, 8, 13, 21, \ldots$

A good example is the Fibonacci matrix

$$
A =
\begin{pmatrix}
1 & 1 \\\\
1 & 0
\end{pmatrix}
$$

The animation below shows what this matrix does to a 2D mesh. In standard coordinates, the mesh shears. But in the eigenvector basis, the same transformation separates cleanly: each family of lines is only scaled along its own direction.

{{< video src="/assets/eigenvectors-the-basis-a-matrix-is-hiding/eigenbasis_geometry.webm" type="video/webm" >}}

This geometric picture is useful because the Fibonacci sequence is built by applying this same matrix again and again. The recurrence

$$
F_{n+1} = F_n + F_{n-1}
$$

can be written as

$$
\begin{pmatrix}
F_{n+1} \\\\
F_n
\end{pmatrix} =
\begin{pmatrix}
1 & 1 \\\\
1 & 0
\end{pmatrix}
\begin{pmatrix}
F_n \\\\
F_{n-1}
\end{pmatrix}
$$

So each Fibonacci step applies $A$ once. Computing later Fibonacci numbers means computing powers of $A$

This is where diagonalization pays off. If $A = PDP^{-1}$, then

$$
A^n = PD^nP^{-1}
$$

Since $D$ is diagonal, raising it to the $n$-th power just means raising each eigenvalue to the $n$-th power:

$$
D^n =
\begin{pmatrix}
\lambda_1^n & 0 \\\\
0 & \lambda_2^n
\end{pmatrix}
$$

For the Fibonacci matrix, solving $Av = \lambda v$ gives the eigenvalues

$$
\varphi = \frac{1+\sqrt{5}}{2}
\qquad
\psi = \frac{1-\sqrt{5}}{2}
$$

with eigenvectors

$$
v_1 =
\begin{pmatrix}
\varphi \\\\
1
\end{pmatrix}
\qquad
v_2 =
\begin{pmatrix}
\psi \\\\
1
\end{pmatrix}
$$

So the matrix can be diagonalized using these two eigendirections. In that basis, taking $n$ Fibonacci steps is no longer repeated shearing. It is just scaling by $\varphi^n$ in one direction and by $\psi^n$ in the other:

$$
A^n =
P
\begin{pmatrix}
\varphi^n & 0 \\\\
0 & \psi^n
\end{pmatrix}
P^{-1}.
$$

Working this out gives Binet's formula:

$$
F_n = \frac{\varphi^n - \psi^n}{\sqrt{5}}
$$

Since $|\psi| < 1$, the term $\psi^n$ becomes negligible as $n$ grows. That is why the Fibonacci numbers grow essentially like

$$
\frac{\varphi^n}{\sqrt{5}}
$$

and why the ratio $F_{n+1}/F_n$ converges to $\varphi$.

## Example 2: Phase portrait of a linear system


## Example 3: Image compression via SVD


## Conclusion

