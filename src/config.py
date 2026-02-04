
from dataclasses import dataclass

@dataclass(frozen=True)
class SimConfig:
    spec_path: str = "variables.json"
    n_initial: int = 20000
    n_final: int = 1000
    global_bad_rate: float = 0.04
    seed: int = 42
    output_dir: str = "output"
    output_basename: str = "pd_simulated"
    # If True, downsample with stratification on Y_final to keep bad rate stable.
    stratified_downsample: bool = True
