from setuptools import setup, find_packages

setup(
    name="custom_loss",
    version="0.1",
    packages=find_packages(),
    # package_data={"custom_loss": ["*.so"]},  # include .so files in the package
    include_package_data=True,
)
