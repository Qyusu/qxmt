# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../qxmt"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qxmt"
copyright = "2024, kenya-sk"
author = "kenya-sk"
release = "0.5.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser"]
autodoc_typehints = "description"
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
# exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ignore warnings about references
suppress_warnings = [
    "ref.python",
]

nitpick_ignore = [("myst", "ref1"), ("myst", "ref2")]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/Qyusu/qxmt",
    "use_repository_button": True,
}

language = "en"
locale_dirs = ["locale/"]
gettext_compact = False
