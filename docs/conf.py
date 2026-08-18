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
    "smudgy.core._cpp_functions_ext",  # needed otherwise API docs fail to build
    # mpi4py and taichi are heavy/native dependencies that are prone to
    # failing to import (missing system MPI runtime, no GPU/display, etc.)
    # in RTD's minimal build container. Mocking them means autodoc never
    # needs a real, working import of them just to read docstrings/signatures.
    "mpi4py",
    "mpi4py.MPI",
    "taichi",
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
html_js_files = ["custom.js",]
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

           "font-stack": "'IBM Plex Sans', -apple-system, sans-serif",
           #"font-stack--headings": "'Roboto', -apple-system, sans-serif",
           "font-stack--monospace": "'Inconsolata', monospace",

           # Match custom.css's code/inline-code font-size bump on the
           # autodoc-generated API signature blocks (dl.py .sig), which use
           # a separate Furo variable that custom.css's overrides don't reach.
           "api-font-size": "1.05em",

           # Keep visited links identical to normal links (no purple).
           "color-link--visited": "var(--color-brand-content)",
           "color-link--visited--hover": "var(--color-brand-content)",
           "color-link-underline--visited": "var(--color-background-border)",
           "color-link-underline--visited--hover": "var(--color-foreground-border)",

           # Backgrounds -- both are native Furo variables (not custom).
           # Uncomment + edit to override; shown values are Furo's own
           # light-mode defaults.
           #"color-background-primary": "#fff",     # main site (article/content) background
           #"color-sidebar-background": "#f8f9fb",  # left sidebar background

           # Text colors -- native Furo variables (not custom).
           # color-sidebar-link-text--top-level defaults to color-brand-primary
           # (currently "black" above) unless set separately here.
           #"color-foreground-primary": "#000",             # main site body text
           #"color-sidebar-link-text": "#5a5c63",            # sidebar nav links (nested)
           #"color-sidebar-link-text--top-level": "black",   # sidebar nav links (top-level)

           # Background flashed behind a heading/table/footnote/etc. when you
           # jump to it via a TOC link or #anchor (native Furo variable, not
           # custom). Shown value is Furo's own light-mode default (yellow).
           "color-highlight-on-target": "#6494ED40",

           #"color-code-background": "#C8D3E9",
           #"color-code-foreground": "#7d5050",
    },
       "dark_css_variables": {
           "color-brand-primary": "black",       # TODO: tune in dark-theme pass
           "color-brand-content": "#7EB4F9",      # TODO: tune in dark-theme pass
           
           "font-stack": "'IBM Plex Sans', -apple-system, sans-serif",
           #"font-stack--headings": "'Roboto', -apple-system, sans-serif",
           "font-stack--monospace": "'Inconsolata', monospace",

           "api-font-size": "1.05em",

           # Keep visited links identical to normal links (no purple).
           "color-link--visited": "var(--color-brand-content)",
           "color-link--visited--hover": "var(--color-brand-content)",
           "color-link-underline--visited": "var(--color-background-border)",
           "color-link-underline--visited--hover": "var(--color-foreground-border)",

           # Backgrounds -- native Furo variables (not custom). Shown
           # values are Furo's own dark-mode defaults; TODO: tune in
           # dark-theme pass.
           "color-background-primary": "#191921",  # main site (article/content) background
           "color-sidebar-background": "#191921",  # left sidebar background

           # Sidebar search box -- defaults to color-background-secondary,
           # which is untouched here and so no longer matched the sidebar
           # once color-sidebar-background was overridden above. Reference
           # the variable (not a duplicated hex) so it keeps tracking
           # color-sidebar-background automatically if that's tuned later.
           "color-sidebar-search-background": "var(--color-sidebar-background)",
           "color-sidebar-search-background--focus": "var(--color-sidebar-background)",

           # Text colors -- native Furo variables (not custom). TODO:
           # tune in dark-theme pass.
           "color-foreground-primary": "#D9DEE8",            # main site body text
           "color-sidebar-link-text": "#D9DEE8",             # sidebar nav links (nested)
           "color-sidebar-link-text--top-level": "#D9DEE8",  # sidebar nav links (top-level)

           # Right-hand "On this page" TOC: color of the currently-scrolled-to
           # section. Overridden directly (rather than inheriting
           # color-brand-primary, which is "black" above and was making it
           # unreadable here) so it doesn't drag every other brand-colored
           # element along with it.
           "color-toc-item-text--active": "#D9DEE8",

           # Background flashed behind a heading/table/footnote/etc. when you
           # jump to it via a TOC link or #anchor (native Furo variable, not
           # custom). Shown value is Furo's own dark-mode default (yellow).
           "color-highlight-on-target": "#6494ED40",

           #"color-code-background": "#1e2127",
           #"color-code-foreground": "#e6e6e6",
    },
}
pygments_style = "tango"        # Style for light mode: black default text, colored keywords/strings/comments
pygments_dark_style = "nord-darker"  # Style for dark mode (Furo specific)
