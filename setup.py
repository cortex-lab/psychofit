from setuptools import setup
from pathlib import Path

CURRENT_DIRECTORY = Path(__file__).parent.absolute()

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='Psychofit',
    version='1.0.0-r0',
    python_requires='>=3.7',
    description='A module for fitting 2AFC psychometric data',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matteo Carandini & Miles Wells',
    url='https://github.com/cortex-lab/psychofit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=['numpy', 'scipy'],
    py_modules=['psychofit']
)
