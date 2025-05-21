import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../tuner/sampling'))

project = 'mytuner'
copyright = '2022, gsunwu@163.com'
author = 'gsunwu@163.com'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "myst_parser"
]

templates_path = ['_templates']

exclude_patterns = []

html_theme = 'alabaster'

html_static_path = ['_static']
