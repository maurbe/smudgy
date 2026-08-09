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
autodoc_mock_imports = [
    "smudgy.core._cpp_functions_ext", # needed otherwise API docs fail to build
]

nb_execution_mode = "off"
nb_execution_in_temp = True
nb_render_image_options = {"width": "600px", "align": "center"}

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

# html_logo = "_static/test.png"
html_theme = "furo"
html_css_files = ["custom.css",]
html_static_path = ["_static"]
html_theme_options = {

    # repository information
    "source_repository": "https://github.com/maurbe/smudgy/",
    "source_branch": "main",
    "source_directory": "docs/",
    #"edit_button_type": "edit",

    "sidebar_hide_name": True,

       "light_css_variables": {
           "color-brand-primary": "black",
           "color-brand-content": "#3352CC",

           "font-stack": "'Roboto', -apple-system, sans-serif",
           "font-stack--headings": "'Roboto', -apple-system, sans-serif",
           "font-stack--monospace": "'Inconsolata', monospace",

           #"color-code-background": "#282c34",
           #"color-code-foreground": "#e6e6e6",
    },
       "dark_css_variables": {
           #"color-code-background": "#1e2127",
           #"color-code-foreground": "#e6e6e6",
    },
}
pygments_style = "catppuccin-latte"        # Style for light mode (dark code panel now used in both modes)
pygments_dark_style = "catppuccin-mocha"  # Style for dark mode (Furo specific)
