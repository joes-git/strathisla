from setuptools import setup, find_packages

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "spey.backend.plugins": [
            "strathisla.full_nuisance_parameters = strathisla.nuisance_parameters:FullNuisanceParameters",
            "strathisla.simple_multivariate_gaussian_eft = strathisla.eft:SimpleMultivariateGaussianEFT",
            "strathisla.multivariate_gaussian_scaled_covariance_eft = strathisla.eft:MultivariateGaussianCovarianceScaledEFT",
        ]
    },
    install_requires=["spey"],
)