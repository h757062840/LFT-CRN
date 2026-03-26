### 3. `LFT-CRN-neb.py` Documentation (`docs/README_LFT-CRN-neb.md`)

```markdown
# Transition State Search Workflow (`LFT-CRN-neb.py`)

The `LFT-CRN-neb.py` script manages iterative Nudged Elastic Band (NEB) calculations to locate transition states and reaction pathways.

## Purpose

Finding transition states using standard DFT is computationally expensive. This module uses MLIPs to pre-relax the NEB images across the reaction coordinate. The ML-relaxed images are then evaluated using VASP to provide accurate forces and energies, which are subsequently used to train the MLIP.

## Execution

The script requires a specific configuration file defining the NEB parameters and multi-stage force thresholds.

```bash
python LFT-CRN-neb.py --config ../utils/config_neb.yaml