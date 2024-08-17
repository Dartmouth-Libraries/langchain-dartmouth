import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.abspath("../src/langchain_dartmouth/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "langchain_dartmouth"
copyright = "by Simon Stone for Dartmouth College Libraries under Creative Commons CC BY-NC 4.0 License"
author = "Simon Stone"
release = version("langchain_dartmouth")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.coverage"]
autodoc_typehints = "description"
autodoc_inherit_docstrings = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = html_static_path[0] + "/img/favicon.png"
html_logo = html_static_path[0] + "/img/logo.png"
html_css_files = [
    "css/custom.css",
]
