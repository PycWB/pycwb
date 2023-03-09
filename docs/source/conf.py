# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyBurst'
copyright = '2023, Yumeng Xu'
author = 'Yumeng Xu'
release = '0.10.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ["pyburst/vendor/*", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- external links ----------------------------------------------------------

intersphinx_mapping = {
    'gwpy': ('https://gwpy.github.io/docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pycbc': ('https://pycbc.org/pycbc/latest/html/', None),
}
