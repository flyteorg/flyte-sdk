import numpy as np
import pandas as pd


def generate_data(columns: list[str], seed: int = 42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.random(size=(20, len(columns))), columns=columns)
    return data
