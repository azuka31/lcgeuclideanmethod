# LCG T-TEST USING EUCLIDEAN METHOD
# ---------------------------------
# Advanced Analytics and Growth Marketing Telkomsel
# -------------------------------------------------
# Project Supervisor : Rizli Anshari
# Writer             : Azka Rohbiya Ramadani, Demi Lazuardi, Muhammad Gilang

from setuptools import find_packages, setup

with open('README.md','r') as f:
    long_description = f.read()
setup(
    author="Azka Rohbiya Ramadan",
    author_email='azkarohbiya@gmail.com',
    name='lcgeuclideanmethod',
    description="Determining LCG population from campaign takers/non by calculating Euclidean Method and ttest",
    version="0.1.0",
    long_description=long_description,
    packages=find_packages(include=['lcgttest','lcgttest.*']),
    install_requires=[
        'pandas>=1.1.5',
        'numpy>=1.18.5',
        'scipy>=1.2.0',
        'matplotlib>=3.1.0',
        'statsmodels>=0.8.0'],
    python_requires='>=3.1' # better in 3.9.9
)
