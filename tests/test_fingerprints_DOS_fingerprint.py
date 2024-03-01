import numpy as np

from madas.fingerprints import DOSFingerprint

def test_generation():
    x = np.linspace(-10,5,100)
    y = np.sin(x)
    grid = DOSFingerprint.get_default_grid()
    grid.e_ref=0
    grid.cutoff_min = -10
    grid.cutoff_max = 5
    grid.grid_type = "uniform"
    fp = DOSFingerprint(grid_id=grid.get_grid_id()).calculate(x, y, convert_data=None)
    assert fp.data["fingerprint"]["grid_id"] == grid.get_grid_id(), "did not set correct grid id"
    assert fp.get_similarity(fp) == 1.0, "Similarity to self is not 1"