import pytest
from pathlib import Path
from house_prices_regression_model.processing.data_manager import load_dataset
from house_prices_regression_model.config.core import DATASET_DIR, FILE_NAME_DATA_TEST

@pytest.fixture()
def sample_test_data():
    return load_dataset(data_path=Path(f"{DATASET_DIR}/{FILE_NAME_DATA_TEST}"))
