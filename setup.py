from setuptools import setup, find_packages

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "spey.backend.plugins": [
            "contur.full_histogram_likelihood = contur_likelihood.likelihoods:ConturHistogram"
        ]
    }
)