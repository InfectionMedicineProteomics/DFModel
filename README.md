# DFModel

This package provides programmatic access to digital family modelling

# Installation

The package can be installed using ```pip``` and installing from source from this github repository

```commandline
pip install git+git@github.com:InfectionMedicineProteomics/DFModel.git
```

# Usage

A full examples for usage is available in the notesbook provided in this repository

# Example

A digital family model can be easily built using the following code as an example

```python
from dfmodel import DigitalFamilyBinary

df_estimator = DigitalFamilyBinary()

df_estimator.fit(X, features=selected_features)

preditions = df_estimator.predict(
    X_test,
    features=selected_features,
    target_column="value"
)
```

In this case ```X``` is the database used as the search space and ```X_test``` is the data that you want to make predictions for.

```selected_features``` are the features to use as the search features for this model.

```target_column``` is the value that you wish to model.

