import pandas as pd
import numpy as np
import pytest

from dfmodel import DigitalFamilyBinary


def create_sample_data(n=50):
    """Create a sample DataFrame with two feature columns and a binary target."""
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.rand(n),
        'feature2': np.random.rand(n),
        'target': np.random.randint(0, 2, size=n)
    })
    return data

def test_fit():
    """Test that the fit method correctly stores the data."""
    data = create_sample_data()
    model = DigitalFamilyBinary(bootstrap_iterations=10)
    model.fit(data)
    pd.testing.assert_frame_equal(model.data_, data)

def test_predict_return_type_and_length():
    """Test that predict returns a pandas Series with the same length as the input."""
    data = create_sample_data(100)
    model = DigitalFamilyBinary(bootstrap_iterations=10, n_neighbors=5)
    model.fit(data)
    predictions = model.predict(data, feature_columns=['feature1', 'feature2'], target_column='target')
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(data)

def test_predict_values_range():
    """Test that all prediction values are within the [0, 1] range."""
    data = create_sample_data(100)
    model = DigitalFamilyBinary(bootstrap_iterations=10, n_neighbors=5)
    model.fit(data)
    predictions = model.predict(data, feature_columns=['feature1', 'feature2'], target_column='target')
    # All probabilities should be between 0 and 1.
    assert predictions.between(0, 1).all()

def test_predict_internal_attributes():
    """Test that the bootstrap_columns and bootstrap_results_df attributes are set properly."""
    data = create_sample_data(50)
    model = DigitalFamilyBinary(bootstrap_iterations=10, n_neighbors=3)
    model.fit(data)
    _ = model.predict(data, feature_columns=['feature1', 'feature2'], target_column='target')
    # Check that bootstrap_columns is a list of the expected length.
    assert isinstance(model.bootstrap_columns, list)
    assert len(model.bootstrap_columns) == 10
    # Check that bootstrap_results_df is a DataFrame with matching columns.
    assert isinstance(model.bootstrap_results_df, pd.DataFrame)
    assert list(model.bootstrap_results_df.columns) == model.bootstrap_columns

def test_predict_without_fit():
    """Test that predict raises an error if fit was not called."""
    data = create_sample_data(20)
    model = DigitalFamilyBinary(bootstrap_iterations=5)
    with pytest.raises(AttributeError):
        # Since model.data_ is None, calling predict should raise an AttributeError.
        model.predict(data, feature_columns=['feature1', 'feature2'], target_column='target')

def test_predict_invalid_feature_column():
    """Test that passing an invalid feature column name raises a KeyError."""
    data = create_sample_data(50)
    model = DigitalFamilyBinary(bootstrap_iterations=5)
    model.fit(data)
    with pytest.raises(KeyError):
        # Using a nonexistent feature should raise a KeyError when sampling the data.
        model.predict(data, feature_columns=['nonexistent_feature'], target_column='target')

def test_predict_invalid_target_column():
    """Test that passing an invalid target column name raises a KeyError."""
    data = create_sample_data(50)
    model = DigitalFamilyBinary(bootstrap_iterations=5)
    model.fit(data)
    with pytest.raises(KeyError):
        # Using a nonexistent target column should raise a KeyError.
        model.predict(data, feature_columns=['feature1', 'feature2'], target_column='nonexistent_target')