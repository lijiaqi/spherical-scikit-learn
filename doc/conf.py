# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Spherical-Scikit-Learn"
copyright = "Jiaqi Li"
author = "Jiaqi Li"
release = "0.1.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
master_doc = "index"
templates_path = ["_templates"]
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
todo_include_todos = True

extensions.append("sphinx_rtd_theme")
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 5,
}

extensions.append("autoapi.extension")
extensions.append("sphinx.ext.napoleon")
autoapi_type = "python"
autoapi_dirs = ["../spsklearn"]
autoapi_ignore = ["*/test_*.py"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # "special-members",
    "show-inheritance-diagram",
    "imported-members",
]
