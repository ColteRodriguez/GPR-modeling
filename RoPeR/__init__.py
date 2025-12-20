# myproject/__init__.py
from importlib.metadata import version

REQUIRED = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "matplotlib": "3.10.7"
}

def resolve_dependency_conf():
    print("installing ___")
    # Installation code here

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
        should_resolve = input("would you like to proceed with installing the correct versions? (Y/N)")
        if should_resolve == "Y":
            resolve_dependency_conf()
        else:
            print("Process exited without resolving conflicts")

check_dependencies()

# optionally expose submodules
from .bgr import *
from .bandpass import *
from .gain import *
from .prestitch import *