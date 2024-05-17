# Doxyfile 1.9.8

#---------------------------------------------------------------------------
# Project related configuration options
#---------------------------------------------------------------------------
DOXYFILE_ENCODING      = UTF-8
PROJECT_NAME           = NEML2
PROJECT_NUMBER         = 1.4.0
OUTPUT_DIRECTORY       = build
TOC_INCLUDE_HEADINGS   = 3

#---------------------------------------------------------------------------
# Build related configuration options
#---------------------------------------------------------------------------
EXTRACT_ALL            = YES
CASE_SENSE_NAMES       = YES
HIDE_SCOPE_NAMES       = YES
SHOW_USED_FILES        = NO
SHOW_FILES             = NO
SHOW_NAMESPACES        = NO
LAYOUT_FILE            = /home/thu/projects/neml2/doc/config/DoxygenLayout.xml

#---------------------------------------------------------------------------
# Configuration options related to warning and progress messages
#---------------------------------------------------------------------------
WARN_NO_PARAMDOC       = YES

#---------------------------------------------------------------------------
# Configuration options related to the input files
#---------------------------------------------------------------------------
INPUT                  = /home/thu/projects/neml2/README.md \
                         /home/thu/projects/neml2/src \
                         /home/thu/projects/neml2/include \
                         /home/thu/projects/neml2/doc/content \
                         /home/thu/projects/neml2/doc/content
FILE_PATTERNS          = *.cxx \
                         *.h \
                         *.md
RECURSIVE              = YES
IMAGE_PATH             = /home/thu/projects/neml2/doc/content/asset
EXCLUDE_SYMBOLS        = neml2::*internal neml2::*details
USE_MDFILE_AS_MAINPAGE = /home/thu/projects/neml2/README.md

#---------------------------------------------------------------------------
# Configuration options related to the alphabetical class index
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Configuration options related to the preprocessor
#---------------------------------------------------------------------------
MACRO_EXPANSION        = YES

#---------------------------------------------------------------------------
# Configuration options related to the dot tool
#---------------------------------------------------------------------------
HAVE_DOT               = YES
COLLABORATION_GRAPH    = NO
INCLUDE_GRAPH          = NO
INCLUDED_BY_GRAPH      = NO

#---------------------------------------------------------------------------
# Configuration options related to the HTML output
#---------------------------------------------------------------------------
GENERATE_HTML          = NO

#---------------------------------------------------------------------------
# Configuration options related to the LaTeX output
#---------------------------------------------------------------------------
GENERATE_LATEX         = NO
#---------------------------------------------------------------------------
# Configuration options related to the HTML output
#---------------------------------------------------------------------------
GENERATE_HTML          = YES
DISABLE_INDEX          = NO
FULL_SIDEBAR           = NO
HTML_OUTPUT            = html
HTML_EXTRA_STYLESHEET  = /home/thu/projects/neml2/_deps/doxygen-awesome-css-src/doxygen-awesome.css\
                         /home/thu/projects/neml2/_deps/doxygen-awesome-css-src/doxygen-awesome-sidebar-only.css
HTML_COLORSTYLE        = LIGHT
GENERATE_TREEVIEW      = YES
USE_MATHJAX            = YES
MATHJAX_VERSION        = MathJax_3
MATHJAX_FORMAT         = HTML-CSS
MATHJAX_RELPATH        = https://cdn.jsdelivr.net/npm/mathjax@3
MATHJAX_EXTENSIONS     = ams physics boldsymbol
WARN_LOGFILE           = /home/thu/projects/neml2/doc/doxygen.html.log
