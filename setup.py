# Try using setuptools first, if it's installed
from setuptools import setup

# define all packages for distribution
packages = [
    'gpvolve',
]

setup(name='gpvolve',
      version='0.1.0',
      description='A Python library for extracting evolutionary trajectories from large genotype-phenotype maps.',
      author='Zach Sailer',
      author_email='zachsailer@gmail.com',
      url='https://github.com/harmslab/gpvolve',
      packages=packages,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
      ],
      zip_safe=False)
