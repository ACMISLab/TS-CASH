import os
import sys
from setuptools import setup


# Raise warnings if system version is not greater than 3.5
if sys.version_info < (3, 6):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. Pynisher requires Python '
        '3.6 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )



HERE = os.path.abspath(os.path.dirname(__file__))

setup(
    name='autosklearn',
    version="0.15.0",
    packages=['autosklearn'],
    extras_require={},
    author="Stefan Falkner, Christina Hernandez-Wunsch, Samuel Mueller and Matthias Feurer and Francisco Rivera",
    author_email="feurerm@informatik.uni-freiburg.de",
    description="A small Python library to limit the resources used by a function by executing it inside a subprocess.",
    long_description_content_type='text/markdown',
    include_package_data=False,
    keywords="resources",
    license="MIT",
    url="https://github.com/automl/pynisher",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)