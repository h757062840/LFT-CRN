# Local-Fine-Tuning framework for Catalytic Reaction Network exploration(LFT-CRN) 

This repository provides an automated active learning workflow coupling Machine Learning Interatomic Potentials (MLIPs) with Density Functional Theory (DFT) calculations. It is designed to perform structural relaxations and transition state searches for surface reactions (e.g., catalysis).

The framework iteratively refines atomic structures using MLIPs, verifies the geometries and energies using first-principles calculations (VASP), and fine-tunes the underlying potential using the accumulated DFT data via `dpdata` and ASE.

## Repository Structure

* `LFT-CRN-opt.py`: Main driver for intermediate structural optimizations (Initial/Final States).
* `LFT-CRN-neb.py`: Main driver for Nudged Elastic Band (NEB) transition state searches.
* `utils/`: Directory containing required VASP inputs, configuration YAMLs, and ML training scripts.
* `docs/`: Additional documentation.
    * [Structural Optimization Guide](LFT-OPT/readme.md)
    * [Transition State Search Guide](LFT-NEB/readme.md)

## Prerequisites

* **Software**: VASP (executable `vasp_std`), VASPKIT, Slurm Workload Manager.
* **Python Packages**: `ase`, `fairchem`, `dpdata`, `numpy`, `torch`, `pyyaml`.

## Core Logic

The workflows in this repository follow a cyclic active learning approach:
1.  **ML Drive**: Relax geometries or NEB images using the current MLIP checkpoint.
2.  **DFT Verification**: Submit the ML-relaxed structures to VASP for high-accuracy evaluation.
3.  **Data Conversion**: Parse VASP `OUTCAR` outputs into an ASE database format.
4.  **Model Fine-tuning**: Train the MLIP on the newly accumulated dataset to produce an updated checkpoint for the next iteration.

For detailed instructions on each module, please refer to their respective documentation files.

The MLIP model and DFT datasets can be download in ZENODO repository https://zenodo.org/records/17303850 
