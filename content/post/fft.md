+++
title = "TODO"
date = 2025-11-22
tags = ["todo"]
+++

{{< toc >}}


## Introduction 

## Signals Are Sums of Oscillations
Every signal—no matter how complex—can be expressed as a weighted sum of pure sinusoids at different frequencies.

In music this obvious (chords)

but ponder for a second, every signal no matter how ocsilating it is or not it can be composed a sum of oscillators.

show animation where sums of sin functions create a square wave

show animation where sums of sin creates noise


Think of it like mixing paint: you can create any color by combining the right amounts of primary colors. Fourier analysis is the reverse process: unmixing pant

## Visual of the fourier transform

show an animation where we take a signal mixture of 2 frequences
map it to the 2d space (spinning phasor in the complex plane)
and show that the center of mass has 3 peaks

## Math behin the fourier transform

fourier transfomr formula

the FT measures how strongly the signal correlates with a complex exponential at each frequency f.

If your signal x(t) contains a sinusoid at frequency f, then multiplying by e^(-j2πft) and integrating gives a large value—they're aligned, so the product accumulates constructively.

If your signal doesn't contain frequency f, the product oscillates positive and negative, averaging to near zero.

this is a correlation 

Also we can see it as aprojection into the basis
why the basis of sin/cos

