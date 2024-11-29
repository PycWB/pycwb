# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pycwb import __version__

project = 'pycWB'
copyright = '2023, Yumeng Xu'
author = 'Yumeng Xu'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ["pycwb/vendor/*", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']

# -- external links ----------------------------------------------------------

intersphinx_mapping = {
    'gwpy': ('https://gwpy.github.io/docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pycbc': ('https://pycbc.org/pycbc/latest/html/', None),
}


# from pycwb.constants import user_parameters_schema
#
# keys = user_parameters_schema['properties'].keys()
# default_values = [user_parameters_schema['properties'][key]['default'] for key in keys]
# descriptions = [user_parameters_schema['properties'][key]['description'] for key in keys]
# types = [user_parameters_schema['properties'][key]['type'] for key in keys]
# maximums = [user_parameters_schema['properties'][key]['maximum'] for key in keys]
# minimums = [user_parameters_schema['properties'][key]['minimum'] for key in keys]
#

import sys
from os.path import basename

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from docutils.parsers.rst import Directive
from docutils import nodes, statemachine

class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [nodes.error(None, nodes.paragraph(text = "Unable to execute python code at %s:%d:" % (basename(source), self.lineno)), nodes.paragraph(text = str(sys.exc_info()[1])))]
        finally:
            sys.stdout = oldStdout

def setup(app):
    app.add_directive('exec', ExecDirective)