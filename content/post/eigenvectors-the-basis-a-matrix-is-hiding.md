+++
title = "Eigenvectors: the basis a matrix is hiding"
date = 2026-05-13
tags = ["linear-algebra", "signal-processing"]

+++

{{< toc >}}

## Introduction

Eigenvectors are defined as follows: a vector $v$ is an eigenvector of matrix $A$ if

$$Av = \lambda v$$

Meaning applying $A$ only scales $v$ by $\lambda$, it does not rotate it. But the real reason eigenvectors matter is what happens when you collect them into a basis. In that basis, $A$ is diagonal. No rotation, no mixing between components. Each direction evolves independently.

TODO

## Example 1: The geometry of a matrix

The equation $Av = \lambda v$ says that $A$ maps $v$ to a scalar multiple of itself: same direction, different length. In every other direction, $A$ rotates and scales at the same time. The eigenvectors are the directions where rotation is absent.

When $A$ has $n$ linearly independent eigenvectors, we can collect them as columns of a matrix $P$ and write $A = PDP^{-1}$, where $D$ is diagonal. In the eigenbasis, $A$ has no off-diagonal terms. Each axis evolves independently of the others.

The animation below shows this on a 2D mesh. In standard coordinates the mesh shears under $A$. In the eigenvector basis, the same transformation only scales each family of lines along its own direction.

{{< video src="/assets/eigenvectors-the-basis-a-matrix-is-hiding/eigenbasis_geometry.webm" type="video/webm" >}}

Diagonalization simplifies computing powers of $A$. In standard coordinates, $A^n$ requires $n$ matrix multiplications. With $A = PDP^{-1}$:

$$A^n = P \begin{pmatrix} \lambda_1^n & 0 \newline 0 & \lambda_2^n \end{pmatrix} P^{-1}$$

The $n$ multiplications reduce to two scalar powers.

**Fibonacci.** The Fibonacci sequence starts from $0$ and $1$, with each term the sum of the two before it:

$$0,\ 1,\ 1,\ 2,\ 3,\ 5,\ 8,\ 13,\ 21,\ \dots$$

The recurrence $F_{n+1} = F_n + F_{n-1}$ can be written as the matrix iteration $v_{n+1} = Av_n$:

$$\begin{pmatrix} F_{n+1} \newline F_n \end{pmatrix} = \begin{pmatrix} 1 & 1 \newline 1 & 0 \end{pmatrix} \begin{pmatrix} F_n \newline F_{n-1} \end{pmatrix}$$

This matrix is diagonalizable. Solving $Av = \lambda v$ gives eigenvalues $\varphi = \frac{1+\sqrt{5}}{2}$ and $\psi = \frac{1-\sqrt{5}}{2}$, with eigenvectors $v_1 = \begin{pmatrix} \varphi \newline 1 \end{pmatrix}$ and $v_2 = \begin{pmatrix} \psi \newline 1 \end{pmatrix}$. Since $v_1$ and $v_2$ are linearly independent, we have $A = PDP^{-1}$ with $D = \mathrm{diag}(\varphi, \psi)$. Raising both sides to the $n$-th power:

$$A^n = P \begin{pmatrix} \varphi^n & 0 \newline 0 & \psi^n \end{pmatrix} P^{-1}$$

Working this out for the Fibonacci matrix gives Binet's formula:

$$F_n = \frac{\varphi^n - \psi^n}{\sqrt{5}}$$

Since $|\psi| < 1$, the $\psi^n$ term vanishes as $n$ grows. The Fibonacci numbers grow like $\varphi^n / \sqrt{5}$, and the ratio $F_{n+1}/F_n$ converges to $\varphi$.

## Example 2: Phase portrait of a linear system

Consider the system:

$$\dot{x} = Ax$$

where $x(t)$ is a trajectory in the plane. In standard coordinates the trajectories spiral and curve across the plane. It is hard to read what $A$ is doing.

[visual: phase portrait]

The eigenvectors are the spine of this portrait. A trajectory that starts along an eigenvector direction stays on it, moving along that line, growing if $\lambda > 0$ and shrinking if $\lambda < 0$. Every other trajectory is a combination of those two directions. The general solution is:

$$x(t) = c_1 e^{\lambda_1 t} v_1 + c_2 e^{\lambda_2 t} v_2$$

In the eigenbasis, the coupled system $\dot{x} = Ax$ splits into two independent scalar equations:

$$\dot{y}_1 = \lambda_1 y_1, \qquad \dot{y}_2 = \lambda_2 y_2$$

Each component evolves on its own. The coupling was not a property of the system. It was a property of the coordinates.

## Example 3: Image compression via SVD

Every matrix $A$ has a decomposition:

$$A = U \Sigma V^T = \sum_{i} \sigma_i u_i v_i^T$$

where $u_i$ and $v_i$ are the eigenvectors of $AA^T$ and $A^TA$ respectively, and $\sigma_i$ are the singular values ordered from largest to smallest.

Each term $\sigma_i u_i v_i^T$ is a rank-1 matrix, a single layer of the image. The singular values tell you how much each layer contributes.

[visual: image reconstructing rank by rank]

Keeping only the top $k$ terms gives the best rank-$k$ approximation. At $k=1$ you see the coarsest structure. At $k=10$ the image is recognizable. At $k=50$ it is nearly complete.

The eigenvectors here are not just the natural coordinates of the matrix. They are a ranking of the information inside it by importance. Compression is the act of discarding the layers that matter least.

## Conclusion

Three domains, one move: find the coordinates where the system is diagonal, solve there, transform back. The complexity was never intrinsic to the system. It was a consequence of looking in the wrong basis.
