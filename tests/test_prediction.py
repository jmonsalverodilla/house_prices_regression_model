import math
import numpy as np
from house_prices_regression_model.predict import make_prediction

#Test function
def test_make_prediction(sample_test_data):

    # Given
    expected_first_prediction_value = 126000
    expected_no_predictions = 1459

    # When
    predictions = make_prediction(input_data=sample_test_data)

    # Then
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=10000)
    assert max(predictions) < 10**6
    assert min(predictions) > 10*3



