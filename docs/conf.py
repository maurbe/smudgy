"""Sphinx configuration for library documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# Fetch project metadata from pyproject.toml
import tomllib

with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), "rb") as f:
    pyproject = tomllib.load(f)

project_info = pyproject["project"]
project = project_info["name"]
author = project_info["authors"][0]["name"]
release = project_info["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
    # "myst_parser", already included in myst_nb
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_member_order = "bysource"
# autodoc_typehints = "description"

nb_execution_mode = "off"
nb_execution_in_temp = True

source_suffix = {
    ".rst": "restructuredtext",
    # ".md": "markdown",
    ".md": "myst-nb",
}

myst_enable_extensions = [
    "colon_fence",
    "html_image",
    "dollarmath",
    "amsmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_css_files = [
    "custom.css",
]
html_theme_options = {
    "sidebar_hide_name": True,
}
html_static_path = ["_static"]
# html_logo = "_static/test.png"
