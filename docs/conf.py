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
    "sphinx_copybutton",
    "sphinx_design",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
