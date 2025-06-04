# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fink-vra'
copyright = '2025, Heloise F. Stevance'
author = 'Heloise F. Stevance'
release = 'dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

extensions = ['sphinx_wagtail_theme']

html_theme = 'sphinx_wagtail_theme'
#html_theme = 'sphinx_book_theme'  # or 'sphinx_rtd_theme' if installed
html_static_path = ['_static']

# This is used by Sphinx in many places, such as page title tags.
project = "Fink VRA"

# These are options specifically for the Wagtail Theme.
html_theme_options = dict(
    project_name = "Fink VRA",
    #favicon = "img/logo.png",
    logo = "img/logo.png",
    logo_alt = "VRA",
    logo_height = 79,
    logo_url = "/",
    logo_width = 75,
    github_url = "https://github.com/HeloiseS/finkvra/docs/",
    header_links = "email | hfstevance@gmail.com , hfstevance.com | https://www.hfstevance.com/",
)

html_last_updated_fmt = "%b %d, %Y"
html_show_sphinx = False
