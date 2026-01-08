from importlib.metadata import version

REQUIRED = {
    "pds4_tools": "1.4",
    "numpy": "1.26.4",
    "pds4_tools": "1.4",
    "scipy": "1.15.3",
    "matplotlib": "3.10.7"
}

def check_dependencies():
    problems = []
    for pkg, needed in REQUIRED.items():
        try:
            installed = version(pkg)
            if installed != needed:
                problems.append(f"{pkg}: installed={installed}, required={needed}")
        except Exception:
            problems.append(f"{pkg}: not installed")

    if problems:
        msg = "Dependency version mismatch:\n" + "\n".join(problems)
        print(f"import Error: {msg}")

check_dependencies()

# optionally expose submodules
from .bgr import *
from .bandpass import *
from .gain import *
from .prestitch import *