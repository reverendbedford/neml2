from pathlib import Path

# The NEML2 root directory
# Also the directory that is running this script
rootdir = Path(".")

# Directory where stub files will be written to
outdir = Path("doc/sphinx/source")

# Look for all header files
file_list = [f.stem for f in rootdir.resolve().glob("include/**/*.h") if f.is_file()]

# Create directory and write stub files
outdir.mkdir(parents=True, exist_ok=True)

template = """{0}
=========================================================================

.. doxygenfile:: {0}.h
"""

for f in file_list:
    ofile = Path(outdir / f).with_suffix(".rst")
    if ofile.is_file():
        print("documentation for " + str(f) + " already exists...skip")
    else:
        with ofile.open("w", encoding="utf-8") as out:
            print("writing stub documentation file for " + f)
            out.write(template.format(f, f))
