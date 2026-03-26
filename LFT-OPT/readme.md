# Structural Optimization Workflow (`LFT-CRN-opt.py`)

The `LFT-CRN-opt.py` script automates the iterative optimization of adsorption structures and surface slabs.

## Purpose

This module aims to find the local minimum energy structures for a given set of initial configurations. It alternates between MLIP-driven BFGS optimization and VASP single-point or short-relaxation steps. The resulting DFT data is appended to a global database to fine-tune the MLIP.

## Execution

The script requires an external configuration file to define paths, parameters, and target directories.

```bash
python LFT-CRN-opt.py --config ../utils/config_opt.yaml