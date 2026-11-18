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
    # mpi4py and quadrants are heavy/native dependencies that are prone to
    # failing to import (missing system MPI runtime, no GPU/display, etc.)
    # in RTD's minimal build container. Mocking them means autodoc never
    # needs a real, working import of them just to read docstrings/signatures.
    "mpi4py",
    "mpi4py.MPI",
    "quadrants",
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

# docs/src holds the notebooks that generate the static wiki figures; they
# are not pages of the docs themselves.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "src/**"]

# Same LaTeX macros as the smudgy paper (MNRAS style), so equations can be
# copied verbatim between paper and docs.
mathjax3_config = {
    "tex": {
        "macros": {
            "mathbfit": [r"\boldsymbol{#1}", 1],  # bold italic vectors, e.g. x
            "mathbfss": [r"\boldsymbol{\mathsf{#1}}", 1],  # bold sans tensors, e.g. H
            "bmath": [r"\boldsymbol{#1}", 1],  # bold Greek, e.g. Sigma
            "upi": r"\pi",
        }
    }
}

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

# Furo variables, identical for light and dark: every colour points at a
# --smudgy-* palette variable from _static/custom.css, which holds the
# light and dark values (same hue, tuned lightness).
_furo_vars = {
    "font-stack": "var(--smudgy-font-body)",
    "font-stack--monospace": "var(--smudgy-font-mono)",
    "api-font-size": "var(--smudgy-code-size)",
    # text and backgrounds
    "color-foreground-primary": "var(--smudgy-fg)",
    "color-foreground-secondary": "var(--smudgy-fg-muted)",
    "color-background-primary": "var(--smudgy-bg)",
    "color-background-border": "var(--smudgy-border)",
    "color-sidebar-background": "var(--smudgy-surface)",
    "color-sidebar-background-border": "var(--smudgy-border)",
    "color-sidebar-search-background": "var(--smudgy-bg)",
    "color-sidebar-search-background--focus": "var(--smudgy-bg)",
    # accent: links, sidebar, toc, anchors
    "color-brand-primary": "var(--smudgy-fg)",
    "color-brand-content": "var(--smudgy-accent)",
    "color-brand-visited": "var(--smudgy-accent)",
    "color-link": "var(--smudgy-accent)",
    "color-link--hover": "var(--smudgy-accent-hover)",
    "color-link--visited": "var(--smudgy-accent)",
    "color-link--visited--hover": "var(--smudgy-accent-hover)",
    "color-link-underline": "var(--smudgy-accent-soft)",
    "color-link-underline--hover": "var(--smudgy-accent-hover)",
    "color-link-underline--visited": "var(--smudgy-accent-soft)",
    "color-link-underline--visited--hover": "var(--smudgy-accent-hover)",
    "color-sidebar-link-text": "var(--smudgy-fg-muted)",
    "color-sidebar-link-text--top-level": "var(--smudgy-fg)",
    "color-sidebar-item-background--hover": "var(--smudgy-accent-soft)",
    "color-sidebar-item-expander-background--hover": "var(--smudgy-accent-soft)",
    "color-toc-item-text--active": "var(--smudgy-accent)",
    "color-toc-item-text--hover": "var(--smudgy-accent-hover)",
    "color-highlight-on-target": "var(--smudgy-accent-target)",
    # code
    "color-code-background": "var(--smudgy-code-bg)",
    "color-code-foreground": "var(--smudgy-code-fg)",
    "color-inline-code-background": "var(--smudgy-code-bg)",
    "color-api-background": "var(--smudgy-code-bg)",
    "color-api-background-hover": "var(--smudgy-accent-soft)",
    "color-api-name": "var(--smudgy-accent)",
    "color-api-pre-name": "var(--smudgy-fg-muted)",
}
html_theme_options = {
    "source_repository": "https://github.com/maurbe/smudgy/",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": True,
    "light_css_variables": _furo_vars,
    "dark_css_variables": _furo_vars,
    "light_logo": "smudgy_logo_light.svg",   # relative to _static/
    "dark_logo": "smudgy_logo_dark.svg",

}
# Catppuccin latte / mocha: same token hues in both modes (furo uses the
# dark style in dark mode). Bold/italic is stripped in custom.css.
pygments_style = "catppuccin-latte"
pygments_dark_style = "catppuccin-mocha"

# --- Theme lab: SMUDGY_THEME_LAB=1 sphinx-build ... ----------------------
# Floating picker (docs/_lab) to try code styles, accent, fonts and sizes
# live; "Copy config" exports the choice. Not part of normal builds.
if os.environ.get("SMUDGY_THEME_LAB"):
    html_static_path.append("_lab")
    html_css_files += ["lab_pygments.css", "lab.css"]
    html_js_files.insert(0, "lab.js")
