# Neural Spectra — mathematical model

Author: Paweł Majsterek  
Status: reproducible computational research model  
Canonical directory: `neural-spectra/`

## 1. Research question

The experiment asks whether neural-network training reorganizes the global
spectrum of a weight matrix while preserving the local finite-size
correlation class of neighboring eigenvalue gaps.

The analysis intentionally separates two scales:

1. **global spectral geometry** — concentration and overall shape of the
   spectrum;
2. **local spectral correlations** — statistics of consecutive neighboring
   eigenvalue gaps.

The experiment also separates learning a structured target from memorizing
random labels.

## 2. Network

For hidden width

[
N \in \{96,256,512\},
]

the multilayer perceptron has architecture

[
64 \rightarrow N \rightarrow N \rightarrow 2.
]

The hidden activation is

[
\phi(z)=\tanh z,
]

and the output layer produces two linear logits.

For each affine layer, weights are initialized independently as

[
W_{ij}\sim\mathcal N\left(0,\frac{1}{n_{\mathrm{in}}}\right),
]

with zero biases.

The main matrix analyzed throughout the study is the middle hidden-layer
weight matrix

[
W_2\in\mathbb R^{N\times N}.
]

## 3. Synthetic task

Inputs are sampled as

[
x\sim\mathcal N(0,I_{64}).
]

The binary label is

[
y=\mathbf 1\left[
x_1+0.8x_2x_3+0.5\sin(x_4)>0
\right].
]

Dataset sizes:

- train: 2048,
- validation: 512,
- test: 1024.

A fixed dataset seed is used across conditions.

## 4. Training

The objective is two-class softmax cross-entropy.

Adam parameters:

[
\eta=10^{-3},\qquad
\beta_1=0.9,\qquad
\beta_2=0.999,\qquad
\epsilon=10^{-8}.
]

Batch size: 128.  
Training epochs: 50.  
Checkpoints:

[
0,1,2,5,10,20,50.
]

The experiment is run for 10 network initialization seeds.

Two training conditions are used:

- **true** — original training labels;
- **shuffled** — a fixed permutation of training labels.

The validation and test labels remain unshuffled.

## 5. Spectral object

At each checkpoint the Gram matrix is constructed:

[
G=W_2^\top W_2.
]

Its eigenvalues are ordered

[
0\le \lambda_1\le\lambda_2\le\cdots\le\lambda_N.
]

Because (G) is a real Gram matrix, the natural global random-matrix reference
is real Wishart/Laguerre rather than GOE.

## 6. Global observables

The primary global observables are:

### Stable rank

[
r_s(W_2)=
\frac{\|W_2\|_F^2}{\|W_2\|_2^2}.
]

For finite-size comparison the normalized quantity is

[
\frac{r_s(W_2)}{N}.
]

### Spectral norm

[
\|W_2\|_2=\sqrt{\lambda_{\max}(G)}.
]

### Normalized trace

[
\frac{\operatorname{tr}G}{N}.
]

### Leading-eigenvalue share

[
\frac{\lambda_{\max}(G)}
{\operatorname{tr}G}.
]

The condition number is auxiliary only because the square-Wishart hard edge
makes the smallest eigenvalue unstable.

## 7. Local gap-ratio statistic

Let

[
s_i=\lambda_{i+1}-\lambda_i.
]

For consecutive gaps define

[
\widetilde r_i=
\frac{\min(s_i,s_{i+1})}
{\max(s_i,s_{i+1})}.
]

This statistic is dimensionless and does not require spectral unfolding.

Only the central 50% of ordered eigenvalues is used.

For widths 96, 256 and 512, the corresponding numbers of valid
(widetilde r) values are 46, 126 and 254.

Per network seed we retain:

- mean (widetilde r),
- median (widetilde r),
- fraction (widetilde r<0.1),
- fraction (widetilde r<0.2).

The unit of replication is the entire network seed. Individual gap ratios are
not pooled across network seeds for between-network inference.

## 8. Random-matrix controls

Four finite-size local controls are used:

### Poisson

Independent exponential gaps:

[
s_i\sim\operatorname{Exp}(1).
]

This represents absence of level repulsion.

### Real Wishart

[
X_{ij}\sim\mathcal N(0,1/N),
\qquad
G=X^\top X.
]

This is the natural real Gram-matrix reference.

### GOE

For (A_{ij}\sim\mathcal N(0,1)),

[
H=\frac{A+A^\top}{\sqrt{2N}}.
]

### GUE

For independent real Gaussian matrices (A,B), define
(Z=A+iB) and

[
H=\frac{Z+Z^\dagger}{\sqrt{4N}}.
]

The Wishart and GOE local statistics are both beta=1 references and are not
treated as distinguishable by the selected short-spectrum observables.

## 9. Statistical unit and formal inference

For paired comparisons, the ten network seeds define ten paired observations.

The reference implementation includes:

- exact two-sided sign-flip enumeration over all (2^{10}=1024) sign patterns;
- paired percentile bootstrap confidence intervals;
- paired standardized effect size
  [
  d_z=\bar d/s_d;
  ]
- Benjamini-Hochberg false-discovery-rate correction.

These tests describe variability over initialization seeds conditional on the
fixed dataset and training setup. They are not dataset-population inference.

## 10. Interpretation boundary

The model does not assert a physical identity between Riemann zeros and neural
networks.

The relation to the earlier riemann-terj work is methodological: in both cases
the research asks whether a complex collection is more intelligible through
the geometry and statistics of relations between many elements than through a
single element viewed in isolation.

Random-matrix universality is a statistical statement and not evidence of a
shared physical mechanism.
