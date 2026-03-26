# Active Learning Framework for Structural Optimization

This repository provides an automated active learning workflow designed for surface chemistry and catalyst structural optimization. By coupling Machine Learning Interatomic Potentials (MLIPs) with Density Functional Theory (DFT), the framework systematically refines atomic structures, verifies results using first-principles calculations, and iteratively fine-tunes the potential to improve predictive accuracy.

## Features

* **Active Learning Loop**: Iteratively performs ML-driven structural relaxation, DFT evaluation, and dynamic fine-tuning.
* **Externalized Configuration**: Uses a customizable `config.yaml` file to control directories, simulation parameters, and constraints without altering the core logic.
* **Job Management**: Native integration with the Slurm workload manager for batch submitting and polling VASP/Fine-tuning tasks.
* **Data Parsing**: Automates INCAR/POSCAR modifications (e.g., handling magnetic moments, selective dynamics via tags) and converts OUTCAR data to ASE databases utilizing `dpdata`.
* **Resilience**: Implements a checkpointing system (`record.txt`) allowing interrupted jobs to resume exactly from the point of failure.

## Prerequisites

Ensure the following software and libraries are installed within your computational environment:

* **Python Libraries**: `ase`, `fairchem`, `dpdata`, `numpy`, `torch`, `yaml`
* **Calculation Software**: 
    * VASP (Vienna Ab initio Simulation Package)
    * VASPKIT
* **System Manager**: Slurm

## Setup

1.  Modify the `config.yaml` file to match your environment and paths:
    ```yaml
    directories:
      - "/path/to/IS1"
      - "/path/to/IS2"
    max_iteration: 4
    fix_atoms: 24
    batch_size: 8
    final_DFT: true
    work_directory: "./"
    utils_dir: "/path/to/utils"
    db_path: "output_database.db"
    record_file: "record.txt"
    origin_checkpoint_path: "/path/to/model/checkpoint.pt"
    ```
    *Note: The `utils_dir` should contain your baseline input files like `INCAR_SCF`, `KPOINTS`, `sub.oc`, `finetune1.yml`, and `main.py`.*

## Usage

Execute the main script using Python. By default, it looks for `config.yaml` in the active directory, but you can explicitly specify the path via arguments.

```bash
python makeopt.py --config config.yaml
