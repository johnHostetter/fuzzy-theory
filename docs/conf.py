"""
Sphinx configuration for fuzzy-theory's documentation.

Built via .github/workflows/documentation.yml:
    sphinx-apidoc -f -o docs/_source/ src
    sphinx-build docs _build

fuzzy-theory is installed editable (pip install -e .) by that workflow before this
runs, so autodoc can import the real `fuzzy` package directly - no sys.path
manipulation needed here.
"""

project = "fuzzy-theory"
copyright_holder = "John Wesley Hostetter"
author = "John Wesley Hostetter"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # parses the existing Google-style Args/Returns docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# autodoc only needs to import each module once to read its docstrings; letting it
# actually import torch/triton/etc. (already installed by the workflow via
# `pip install -e .` + requirements.txt) keeps type hints and defaults rendered
# accurately, so nothing is mocked out here.

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
