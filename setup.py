#!/Users/leandergoldbach/miniconda3/bin/python

from setuptools import setup

# Package meta-data.
NAME = 'evoTPT'
DESCRIPTION = 'A Python API for applying transition path theory to genotype-phenotype maps'
URL = 'https://github.com/lgoldbach/evoTPT'
EMAIL = 'l.d.goldbach@students.uu.nl'
AUTHOR = 'Leander D. Goldbach'
REQUIRES_PYTHON = '>=3.3.0'
VERSION = '0.1'


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=['evoTPT'],
    license='MIT'
    )   