# MAGPIE — MAGnetosheath Particle Interaction Explorer

Python tools for applying the **field–particle correlation (FPC)** technique to
MMS burst-mode data, to measure kinetic-scale energy transfer between
electromagnetic fields and particles in magnetosheath turbulence.
Built on [pyspedas](https://github.com/spedas/pyspedas).

## Validation

MAGPIE reproduces the FPC results of two Nature Communications studies:

- Chen, Klein & Howes (2019), *Evidence for electron Landau damping in space plasma turbulence*
- Afshari et al. (2024), *Direct observation of ion cyclotron damping of turbulence in Earth's magnetosheath plasma*

## Modules

| Module | Purpose |
|---|---|
| `field_particle_correlation.py` | **Main module.** Self-contained FPC pipeline for FPI ion or electron distributions: E + v×B frame transformation, field-aligned basis (fixed mean-field or instantaneous), Butterworth highpass/bandpass filtering, FPI interleave handling, and gyrotropic (v∥, \|v⊥\|) or Cartesian (v∥, v⊥1, v⊥2) velocity-space projections. Returns full-interval or time-resolved (`n_subintervals`) correlations for all three field-aligned E components |
| `fpc_uncertainty.py` | Propagates Poisson counting uncertainties (FPI `disterr`) through the FPC pipeline, following Afshari et al. (2021, JGR) Appendix B. Run after `field_particle_correlation` |
| `lorentz.py` | Standalone E + v×B transformation of the electric field into the mean plasma frame |
| `pressure_strain.py` | Pi-D and pressure dilatation from four-spacecraft gradients (reciprocal-vector method), as an independent energy-conversion diagnostic |
| `current_density.py` | Current density j = n q (v<sub>i</sub> − v<sub>e</sub>) from FPI moments, assuming quasineutrality |
| `boxcar_averager.py` | Boxcar averaging of a time series onto a coarser time grid |

**Notation:** `field_particle_correlation` outputs C′ = q v∥ E∥ δf (and the
perpendicular analogues). The C = −q (v∥²/2) ∂δf/∂v∥ E∥ form can be obtained
from C′ by finite differencing in v∥.

## Quick start

Requires `numpy`, `scipy`, `pyspedas`, and `requests`.
To import the functions, at the top of a notebook run:

```python
import requests

base_url = 'https://raw.githubusercontent.com/Parker20019027/magpie/main/'
files = ['boxcar_averager.py', 'current_density.py',
         'field_particle_correlation.py', 'fpc_uncertainty.py',
         'lorentz.py', 'pressure_strain.py']

for f in files:
    response = requests.get(base_url + f)
    with open(f, 'w', encoding='utf-8') as file:
        file.write(response.text)

from boxcar_averager import boxcar_averager
from current_density import current_density
from field_particle_correlation import field_particle_correlation
from fpc_uncertainty import fpc_uncertainty
from lorentz import lorentz
from pressure_strain import pressure_strain
```

## Author

Jack Parker — PhD, Northumbria University
