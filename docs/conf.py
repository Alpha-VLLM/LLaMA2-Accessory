# -- Project information -----------------------------------------------------
import os
from urllib.request import urlopen
from pathlib import Path

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
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_examples",
    "sphinx_tabs.tabs",
    "sphinx_thebe",
    "sphinx_togglebutton",
    # "sphinxext.opengraph",
    # For the kitchen sink
    "sphinx.ext.todo",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "pst": ("https://pydata-sphinx-theme.readthedocs.io/en/latest/", None),
}
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = 'logo_small.png'
html_title = "LLaMA2-Accessory"
html_copy_source = True
html_favicon = "_static/images/logo_small.svg"
html_last_updated_fmt = ""

html_sidebars = {
    "reference/blog/*": [
        "navbar-logo.html",
        "search-field.html",
        "postcard.html",
        "recentposts.html",
        "tagcloud.html",
        "categories.html",
        "archives.html",
        "sbt-sidebar-nav.html",
    ]
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
nb_execution_mode = "cache"
thebe_config = {
    "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
    "repository_branch": "master",
}

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/Alpha-VLLM/LLaMA2-Accessory",
    "repository_branch": "main",
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     "colab_url": "https://colab.research.google.com/",
    #     "deepnote_url": "https://deepnote.com/",
    #     "notebook_interface": "jupyterlab",
    #     "thebe": True,
    #     # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    # },
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
    # "announcement": (
    #     "⚠️The latest release refactored our HTML, "
    #     "so double-check your custom CSS rules!⚠️"
    # ),
    "logo": {
        "image_dark": "logo_small.png",
        "text": html_title,  # Uncomment to try text with logo
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Alpha-VLLM/LLaMA2-Accessory",
            "icon": "fa-brands fa-github",
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

# sphinxext.opengraph
# ogp_social_cards = {
#     "image": "_static/logo-square.png",
# }


def setup(app):
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

from docutils import nodes
import re
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