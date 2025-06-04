import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # So Sphinx can find your code

project = 'finkvra'
author = 'Heloise F Stevance'
release = 'dev'

extensions = [
    'sphinx.ext.autodoc',      # Auto docstrings from code
    'sphinx.ext.napoleon',     # Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_book_theme'  # or 'sphinx_rtd_theme' if installed
html_static_path = ['_static']