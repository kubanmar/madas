import pytest
from madas.apis.NOMAD_web_API import get_atoms

import numpy as np

@pytest.fixture()
def mock_response():
    data = {
        "archive" : {
            "results" : {
                "properties" : {
                    "structures" : {
                        "structure_original" : {
                            "cartesian_site_positions" : [
                                [0,0,0],
                                [25e-12,25e-12,25e-12]
                            ],
                            "lattice_vectors" : [
                                [1e-10, 0, 0],
                                [0, 1e-10, 0],
                                [0, 0, 1e-10]
                            ],
                            "species_at_sites" : [
                                "C", "Si"
                            ]
                        }
                    }
                }
            },
            "mock_data" : "some_test_data",
            "run" : [
                {"system" : [{
                    "atoms" : {
                        "periodic" : [True, True, True]
                    }
                }]}
            ]
        }
    }
    return data

def test_get_atoms(mock_response):

    at = get_atoms(mock_response)

    assert np.allclose(at.positions, [[0,0,0], [.25,.25,.25]]), "Failed to parse atomic positions"
    assert np.allclose(at.cell, np.eye(3)), "Failed to parse atomic cell"
    assert (at.symbols == ["C", "Si"]).all(), "Failed to parse atomic symbols"

