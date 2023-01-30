# Copyright 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from pathlib import Path
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--modify",
    help="Modify the files to have the correct copyright heading",
    action="store_true",
)
args = parser.parse_args()

extensions = {".h": "//", ".cxx": "//", ".py": "#", ".sh": "#"}
additional_files = {}

exclude_dirs = ["extern"]
exclude_files = []


rootdir = Path(".")


def should_check(path):
    for exclude_dir in exclude_dirs:
        if Path(rootdir) / Path(exclude_dir) in path.parents:
            return False

    if path.name in exclude_files:
        return False

    if path.suffix in extensions:
        return True

    if path.name in additional_files:
        return True

    return False


def generate_copyright_heading(copyright, prefix):
    return prefix + " " + (prefix + " ").join(copyright.splitlines(True))


def has_correct_heading(path, copyright, prefix, modify):
    heading = generate_copyright_heading(copyright, prefix)

    # First check if it has the correct heading
    content = path.read_text()
    correct = content.startswith(heading)

    if not modify:
        return correct

    # Correct the heading
    with path.open("w", encoding="utf-8") as file:
        file.write(heading)
        file.write("\n")
        for line in content.splitlines(True):
            if not line.startswith(prefix):
                file.write(line)

    print("Corrected copyright heading for " + str(path))

    return True


files = subprocess.run(
    ["git", "ls-tree", "-r", "HEAD", "--name-only"], capture_output=True, text=True
).stdout

copyright = Path(rootdir / "LICENSE").read_text()
print("The copyright statement is")
print(copyright)

success = True
for file in files.splitlines():
    file_path = Path(file)
    if should_check(file_path):
        if file_path.suffix in extensions:
            prefix = extensions[file_path.suffix]
        elif file_path.name in additional_files:
            prefix = additional_files[file_path.name]
        else:
            sys.exit("Internal error")

        if not has_correct_heading(file_path, copyright, prefix, args.modify):
            print(file)
            success = False

if success:
    print("All files have the correct copyright heading")
else:
    sys.exit("The above files do NOT contain the correct copyright heading")
