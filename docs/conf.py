# -- Project information -----------------------------------------------------
import os
from sphinx.application import Sphinx
from urllib.request import urlopen
from pathlib import Path
from docutils import nodes
import re

project = 'LLaMA2-Accessory'
copyright = 'copyright'
author = 'author'
release = ''
# language = "fr"  # For testing language translations

master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.rediraffe",
    # disabled due to https://github.com/mgaitan/sphinxcontrib-mermaid/issues/109
    # "sphinxcontrib.mermaid",
    "sphinxext.opengraph",
    "sphinx_pyscript",
    "sphinx_togglebutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

suppress_warnings = ["myst.strikethrough"]

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3.8", None),
#     "sphinx": ("https://www.sphinx-doc.org/en/master", None),
#     "pst": ("https://pydata-sphinx-theme.readthedocs.io/en/latest/", None),
# }

# -- Autodoc settings ---------------------------------------------------
autodoc2_packages = [
    {
        "path": "../accessory",
        "exclude_files": ["main*.py", "engine*.py", "*model/LLM/*"],
        "auto_mode": False,
    }
]
autodoc2_hidden_objects = ["dunder", "private", "inherited"]
autodoc2_replace_annotations = [
    ("re.Pattern", "typing.Pattern"),
    ("markdown_it.MarkdownIt", "markdown_it.main.MarkdownIt"),
]
autodoc2_replace_bases = [
    ("sphinx.directives.SphinxDirective", "sphinx.util.docutils.SphinxDirective"),
]
autodoc2_docstring_parser_regexes = [
    ("myst_parser", "myst"),
    (r"myst_parser\.setup", "myst"),
]
nitpicky = True
nitpick_ignore_regex = [
    (r"py:.*", r"docutils\..*"),
    (r"py:.*", r"pygments\..*"),
    (r"py:.*", r"typing\.Literal\[.*"),
]
nitpick_ignore = [
    ("py:obj", "myst_parser._docs._ConfigBase"),
    ("py:exc", "MarkupError"),
    ("py:class", "sphinx.util.typing.Inventory"),
    ("py:class", "sphinx.writers.html.HTMLTranslator"),
    ("py:obj", "sphinx.transforms.post_transforms.ReferencesResolver"),
]

# -- MyST settings ---------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    # "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]

myst_number_code_blocks = ["typescript"]
myst_heading_anchors = 5
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_enable_checkboxes = True
myst_substitutions = {
    "role": "[role](#syntax/roles)",
    "directive": "[directive](#syntax/directives)",
}

# -- HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = '_static/images/logo.png'
html_title = "LLaMA2-Accessory"
html_copy_source = True
html_favicon = "_static/images/logo.png"
html_last_updated_fmt = ""
html_theme_options = {
    "home_page_in_toc": True,
    "github_url": "https://github.com/Alpha-VLLM/LLaMA2-Accessory",
    "repository_url": "https://github.com/Alpha-VLLM/LLaMA2-Accessory",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
    # "announcement": (
    #     "⚠️The latest release refactored our HTML, "
    #     "so double-check your custom CSS rules!⚠️"
    # ),
    "logo": {
        "image_dark": "_static/images/logo.png",
        "text": html_title,  # Uncomment to try text with logo
    },
    "icon_links": [
        {
            "name": "HuggingFace",
            "url": "https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory",
            "icon": "_static/images/hf.svg",
            "type": "local",
        },
    ],
    # For testing
    # "use_fullscreen_button": False,
    # "home_page_in_toc": True,
    # "extra_footer": "<a href='https://google.com'>Test</a>",  # DEPRECATED KEY
    # "show_navbar_depth": 2,
    # Testing layout areas
    # "navbar_start": ["test.html"],
    # "navbar_center": ["test.html"],
    # "navbar_end": ["test.html"],
    # "navbar_persistent": ["test.html"],
    # "footer_start": ["test.html"],
    # "footer_end": ["test.html"]
}

# html_sidebars = {
#     "reference/blog/*": [
#         "navbar-logo.html",
#         "search-field.html",
#         "postcard.html",
#         "recentposts.html",
#         "tagcloud.html",
#         "categories.html",
#         "archives.html",
#         "sbt-sidebar-nav.html",
#     ]
# }
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# nb_execution_mode = "cache"
# thebe_config = {
#     "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
#     "repository_branch": "master",
# }

# sphinxext.opengraph
# ogp_social_cards = {
#     "image": "_static/logo-square.png",
# }

# -- LaTeX output -------------------------------------------------

latex_engine = "xelatex"

def setup(app: Sphinx):
    # -- To demonstrate ReadTheDocs switcher -------------------------------------
    # This links a few JS and CSS files that mimic the environment that RTD uses
    # so that we can test RTD-like behavior. We don't need to run it on RTD and we
    # don't wanted it loaded in GitHub Actions because it messes up the lighthouse
    # results.
    if not os.environ.get("READTHEDOCS") and not os.environ.get("GITHUB_ACTIONS"):
        app.add_css_file(
            "https://assets.readthedocs.org/static/css/readthedocs-doc-embed.css"
        )
        app.add_css_file("https://assets.readthedocs.org/static/css/badge_only.css")

        # Create the dummy data file so we can link it
        # ref: https://github.com/readthedocs/readthedocs.org/blob/bc3e147770e5740314a8e8c33fec5d111c850498/readthedocs/core/static-src/core/js/doc-embed/footer.js  # noqa: E501
        app.add_js_file("rtd-data.js")
        app.add_js_file(
            "https://assets.readthedocs.org/static/javascript/readthedocs-doc-embed.js",
            priority=501,
        )
    app.add_role('link2repo', autolink())


def autolink():
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        pattern = re.compile("\[(.*?)\]\((.*?)\)")
        match_result = pattern.match(text).groups()
        text = match_result[0]
        url = os.path.join("https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/", match_result[1])
        node = nodes.reference(rawtext, text, refuri=url, **options)
        node['classes'].append("github")
        return [node], []
    return role