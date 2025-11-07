# density_field_properties

Tools for computing and analyzing properties derived from the **dark matter density field** in cosmological simulations.

This repository provides methods and scripts for:
- Building the **density field** from dark matter particles using grid assignment schemes (CIC, TSC, etc.).
- Computing the **tidal tensor** and its eigenvalues/eigenvectors.
- Deriving related quantities such as **tidal anisotropy** and **linear bias**.
- Performing auxiliary analyses on density fields, including smoothing, Fourier transforms, and memory-efficient batching.

---

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/<your-username>/density_field_properties.git
cd density_field_properties
conda env create -f environment.yml
conda activate density_field
````

Alternatively, install it as a local editable package:

```bash
pip install -e .
```

---

## Main dependencies

* `numpy`
* `scipy`
* `h5py`
* `matplotlib`
* `tqdm`
* `numba` *(optional, for JIT acceleration)*

---

## Scientific context

This codebase is designed for studies involving:

* **Assembly bias** and extended HOD models (e.g. [`hod_madrid_py`](https://github.com/computationalAstroUAM/hod_madrid_py)).
* **Tidal tensor** and **anisotropy** in large-scale structure ([Haloscope project](https://arxiv.org/pdf/2410.07361)).
* Applications to **UNIT** ([UNIT simulations](https://ui.adsabs.harvard.edu/abs/2024A%26A...689A..69G/abstract)), **FastPM** ([FastPM GitHub](https://github.com/fastpm/fastpm)), and **CAMELS** ([CAMELS project](https://camels.readthedocs.io/en/latest/)).
* Integration within cosmological survey frameworks such as **Euclid** ([ESA Euclid mission](https://www.euclid-ec.org)) and **DESI** ([DESI collaboration](https://www.desi.lbl.gov/)).
