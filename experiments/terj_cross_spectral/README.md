# TERJ Cross-Spectral Test

This directory contains the new cross-domain experiment connecting the earlier
Riemann work and Neural Spectra at the level of a falsifiable statistical
question.

The project does **not** assume that "Riemann = neural network".

It tests whether

[
[mathrm{Riemann}]_{mathrm{rel}}
stackrel{?}{simeq}
[mathrm{Neural}]_{mathrm{rel}}
]

after symmetry class, scale and finite-size effects have been removed by
domain-specific null models.

## Files

- `PROTOCOL.md` — frozen scientific protocol for Stage A
- `DATA_FORMAT.md` — JSON input contract
- `cross_spectral.py` — reference analysis program
- `example_input.json` — synthetic schema example only, not scientific data

## Scientific order

Stage A: static residual cross-spectrum  
Stage B: finite-size cross-scaling  
Stage C: spectral dynamics  
Stage D: blind identification and out-of-sample transfer

Only Stage A is implemented here.

## Run

```bash
python experiments/terj_cross_spectral/cross_spectral.py \
  --input experiments/terj_cross_spectral/example_input.json \
  --out /tmp/cross_spectral_result.json
```

The example file exists only to verify the implementation. It is not evidence
for or against TERJ.

## Licensing

This experiment is part of the Paweł Majsterek research project. Before reuse,
check the repository license and the licensing terms attached to the relevant
source material. Neural Spectra has its own project-specific license.
