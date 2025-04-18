# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# -- Project information

project = 'SCMG'
author = 'Xingjie Pan'

release = '1.0'
version = '1.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    "sphinx.ext.napoleon",  # for NumPy/Google docstring support
]

autodoc_member_order = "bysource"          # Keeps class/function order consistent with your code
autoclass_content = "both"                 # Show class-level and __init__ docstring
typehints_fully_qualified = False          # Optional: shortens type hints

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Jupyter notebook
nbsphinx_execute = 'never'
