"""For backwards compatibility"""

from __future__ import annotations
import sys
#sys.setrecursionlimit(100000)
import wandb
#replace the API with one you applied for from Wandb
wandb_api_key = '000000'
wandb.login(key=wandb_api_key)

if __name__ == "__main__":
    from fairchem.core._cli import main

    main()
