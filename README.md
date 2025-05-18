# HiddenFoot: Efficient Thermodynamic Model to Reveal Transcription Factor and Nucleosome Binding Patterns on Single DNA Molecules

**HiddenFoot** is a high-performance C++ toolkit for inferring transcription factor (TF) and nucleosome binding from single-molecule methylation data. It implements a probabilistic thermodynamic model that integrates DNA binding motifs (PWMs), methylation state, and binding logic using efficient forward-backward algorithms, SGD, and MCMC.



<div align="center">
  <img src="https://github.com/MolinaLab-IGBMC/HiddenFoot/assets/34145153/d49969d7-ed83-4ad4-aa21-97eabc37ea2d" width="300" height="300">
</div>

---

## 🚀 Features

- Thermodynamic modeling of TF and nucleosome binding on individual DNA molecules
- Support for single-molecule methylation assays (e.g., Fiber-seq)
- Parameter inference using:
  - **SGD (ADAM optimizer)**
  - **MCMC (Metropolis-Hastings)**
- Parallel computation via **OpenMP**
- Motif scanning and PWM support (including reverse complements)
- Methylation simulation engine
- Sparse or dense output for profiles and parameters

---

## 🛠 Installation

### 🔧 Dependencies

- C++ compiler (tested with `g++-mp-12` and `clang++`)
- OpenMP support

### 📦 Build using `make`

A `makefile` is included for convenience. Run:

```bash
make
```

### 💡 Available make targets

```bash
make clang       # Build using clang++ with OpenMP
make gpp         # Build using g++-mp-12 with OpenMP
make clean       # Remove the compiled binary
```

You can also compile manually:

```bash
clang++ -O3 -Xpreprocessor -fopenmp -L/opt/local/lib/libomp/ \
  -I/opt/local/include/libomp/ hiddenfoot.cpp -o hiddenfoot -lomp
```

Or:

```bash
g++-mp-12 -O3 -Xpreprocessor -fopenmp -L/opt/local/lib/libomp/ \
  -I/opt/local/include/libomp/ hiddenfoot.cpp -o hiddenfoot -lomp
```

---

## 📥 Input Files

| Type   | Flag      | Description |
|--------|-----------|-------------|
| Sequence      | `-s` | DNA sequence file (single FASTA-like line) |
| Methylation   | `-m` | Methylation matrix (header = sites, rows = molecules) |
| PWMs          | `-w` | Motif file in custom PWM format |
| Parameters    | `-p` | (Optional) Initial parameters |
| Binding list  | `-b` | (Optional) List of preselected binding sites |
| Output prefix | `-o` | Base name for output files |

---

## 🔧 Command-line Interface

```bash
./hiddenfoot <runmode> [options]
```

### Run modes

- `run`: Fit parameters (SGD + MCMC), compute binding profiles
- `fit`: Fit parameters only (SGD + MCMC)
- `bnd`: Compute binding profiles with given parameters
- `sim`: Simulate synthetic single-molecule profiles

### Example

```bash
./hiddenfoot run -s input.fa -m methylation.txt -w motifs.txt -o output
```

---

## 🧪 Output Files

| File | Description |
|------|-------------|
| `*.states.txt` | List of detected binding site elements |
| `*.probS.txt`  | Matrix of PWM scores per position |
| `*.params_SGD.txt` / `*.params_MCMC.txt` | Fitted parameters |
| `*.hfprofile.*.txt` | Posterior profiles (dense) |
| `*.hfbinding.*.txt` | Posterior profiles (sparse, thresholded) |
| `*.simdata.*.txt` | Simulated single-molecule methylation/binding |

---

## 🧬 Configuration Parameters

You can configure various model and optimization parameters via command-line:

```text
-pseudo        Pseudo-count for PWM smoothing (default: 0.5)
-bgprob        Background nucleotide probability (default: 0.25)
-cutoffwms     PWM match cutoff (default: 0.001)
-nuclen        Length of nucleosome footprint (default: 147)
-padlen        Sequence flank padding (default: 150)
-numepochs     SGD training epochs (default: 100)
-batchsize     Number of molecules per SGD batch (default: 64)
-lrate         Learning rate for SGD (default: 0.01)
-numiter       MCMC iterations (default: 1000)
-sigma         Proposal std for MCMC (default: 0.01)
-numthreads    Number of OpenMP threads (default: 4)
-sparse        Threshold for sparse output (default: 0.0)
```

---

## 📖 Citation

Biophysical Modeling Uncovers Transcription Factor and Nucleosome Binding on Single DNA Molecules
Lasha Dalakishvili, Husain Managori, Anais Bardet, Vera Slaninova, Edouard Bertrand and Nacho Molina
bioRxiv 2025.05.13.653852; doi: https://doi.org/10.1101/2025.05.13.653852

---

## 👩‍🔬 Authors

Developed by **Nacho Molina**  
Molina Lab @ IGBMC: https://molinalab.org](https://www.igbmc.fr/en/recherche/teams/stochastic-systems-biology-of-gene-regulation)  

---

## 🧾 License

This project is licensed under the **MIT License**.

---
