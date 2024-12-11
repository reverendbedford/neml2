# Jupyter notebook examples of the NEML2/pyzag interface

## Setup

To run these notebooks install the packages given in `python/examples/requirements.txt`

```
pip install -r python/examples/requirements.txt
```

after building the NEML2 python package and installing the main set of python requirements in `requirements.txt` in the root directory.

## Version control for Jupyter notebook examples

We track these notebooks with the `nbdime` python tool, installed as part of the package requirements.  The first time you install this for use in a new repository you need to run

```
git-nbdiffdriver config --enable
```

to enable the hooks.