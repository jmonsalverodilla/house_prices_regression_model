#Imports
from house_prices_regression_model.train_pipeline import run_training
from house_prices_regression_model.processing.data_manager import load_pipeline
from house_prices_regression_model.config.core import ROOT, TRAINED_MODEL_DIR,load_config_file,SETTINGS_PATH
from house_prices_regression_model import __version__ as VERSION
from pathlib import Path
import sklearn


#Config files
config = load_config_file(SETTINGS_PATH)
PIPELINE_ARTIFACT_NAME = config["PIPELINE_ARTIFACT_NAME"]

def test_model_save_load():
    """
    Tests for the model saving process
    """
    run_training()

    # =================================
    # TEST SUITE
    # =================================
    # Check the model file is created/saved in the directory
    PATH = ROOT / TRAINED_MODEL_DIR/ f"{PIPELINE_ARTIFACT_NAME}_v{VERSION}.pkl"
    assert Path.exists(PATH)

    # Check that the model file can be loaded properly
    # (by type checking that it is a sklearn linear regression estimator)
    pipeline_file_name = f"{PIPELINE_ARTIFACT_NAME}_v{VERSION}.pkl"
    loaded_model = load_pipeline(file_name=pipeline_file_name)
    assert isinstance(loaded_model, sklearn.pipeline.Pipeline)