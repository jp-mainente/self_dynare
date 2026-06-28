from setuptools import setup, find_packages

setup(
    name="dynare_py",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "pandas>=1.4",
    ],
    python_requires=">=3.9",
)
