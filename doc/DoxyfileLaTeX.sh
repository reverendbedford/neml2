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
# Configuration options related to the LaTeX output
#---------------------------------------------------------------------------
GENERATE_LATEX         = YES
LATEX_CMD_NAME         = pdflatex
EXTRA_PACKAGES         = {amsmath},{lmodern},{physics}
LATEX_HEADER           = /home/thu/projects/neml2/doc/config/ANLReportHeader.tex
LATEX_FOOTER           = /home/thu/projects/neml2/doc/config/ANLReportFooter.tex
LATEX_EXTRA_STYLESHEET = /home/thu/projects/neml2/doc/config/ANLReportExtra.sty
LATEX_HIDE_INDICES     = YES
WARN_LOGFILE           = /home/thu/projects/neml2/doc/doxygen.latex.log

#---------------------------------------------------------------------------
# Configuration options related to the dot tool
#---------------------------------------------------------------------------
MAX_DOT_GRAPH_DEPTH    = 2
