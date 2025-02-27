from setuptools import setup, find_packages  # type: ignore

version = "0.0.1"

install_requires = [
    "pandas",
    "scikit-learn",
    "jupyterlab",
]


setup(
    name="dfmodel",
    author="Aaron Scott",
    author_email="aaron.scott@med.lu.se",
    install_requires=install_requires,
    long_description="Digital family modelling",
    include_package_data=True,
    packages=find_packages(include=["dfmodel", "dfmodel.*"]),
    url="",
    version=version,
)