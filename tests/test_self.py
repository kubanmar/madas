import pytest
import os

def test_self():
    """
    Testing if test locations are well defined.
    """

    from simdatframe._test_data import test_data_path

    assert os.path.exists(os.path.join(test_data_path, '.test_data_file_ref'))
