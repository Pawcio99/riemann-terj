# Neural Spectra

**Author:** Paweł Majsterek

Neural Spectra is a reproducible numerical experiment studying how training
changes the spectrum of the middle weight matrix of a multilayer perceptron.

The central empirical distinction is between:

- strong global spectral reorganization during training, and
- comparatively stable local beta=1 gap statistics.

The project includes a true-label condition, a shuffled-label memorization
control, finite-size widths 96/256/512, and finite-size random-matrix controls.

## Files

- [MODEL.md](MODEL.md) — mathematical definition of the experiment
- [neural_spectra.py](neural_spectra.py) — standalone NumPy reference program
- [LICENSE.md](LICENSE.md) — non-commercial research and attribution license
- [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) — commercial licensing policy
- [CITATION.cff](CITATION.cff) — citation metadata
- [publication](./index.html) — integrated web publication

## Quick start

Python 3.10+ and NumPy are sufficient.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r neural-spectra/requirements.txt
```

Run a small verification experiment:

```bash
python neural-spectra/neural_spectra.py experiment \
  --widths 96 \
  --seeds 2 \
  --epochs 5 \
  --out neural-spectra/example-results.json
```

Run the published geometry at all three widths:

```bash
python neural-spectra/neural_spectra.py experiment \
  --widths 96 256 512 \
  --seeds 10 \
  --epochs 50 \
  --out neural-spectra/network-results.json
```

Generate local random-matrix controls:

```bash
python neural-spectra/neural_spectra.py controls \
  --widths 96 256 512 \
  --realizations 5000 \
  --out neural-spectra/control-results.json
```

The controls command can be computationally expensive at width 512.

## Licensing

The files in this directory use the project-specific
[Neural Spectra Research and Attribution License](LICENSE.md), not the
repository-level MIT license, unless an individual file explicitly says
otherwise.

Non-commercial research use is permitted with attribution.

Commercial use requires a separate written commercial license with Paweł
Majsterek, including an agreed revenue- or profit-sharing mechanism.

## Citation

Please cite the project using [CITATION.cff](CITATION.cff) and retain the
following credit:

> Based in part on research and models developed by Paweł Majsterek.

## Scientific limitation

This repository reports a finite computational experiment. It does not claim a
physical relation between neural networks and the zeros of the Riemann zeta
function, and it does not establish an asymptotic universality theorem.
