import os
import sys
import pkg_resources
import importlib

# List of files provided by Glob (truncated for brevity, but I will read them all in reality)
# To be robust, I will walk the directory instead of hardcoding the Glob output.

def get_all_python_files(root_dir):
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

def extract_imports(file_path):
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    parts = line.split()
                    if len(parts) > 1:
                        # Simple heuristic: take the first part after import/from
                        # e.g. "import numpy as np" -> "numpy"
                        # e.g. "from langchain_core.messages import ..." -> "langchain_core"
                        module_name = parts[1].split('.')[0]
                        imports.add(module_name)
    except Exception as e:
        # print(f"Error reading {file_path}: {e}")
        pass
    return imports

def check_imports():
    root_dir = os.getcwd()
    all_files = get_all_python_files(root_dir)
    
    all_imports = set()
    for f in all_files:
        all_imports.update(extract_imports(f))
    
    # Filter out local modules (directories in current project)
    local_modules = {d for d in os.listdir(root_dir) if os.path.isdir(d)}
    # Also add standard library modules (approximate)
    # Actually, importlib.util.find_spec is better.
    
    missing_packages = []
    
    print("Checking imports...")
    for module in sorted(list(all_imports)):
        if module in local_modules:
            continue
            
        # Try to import
        try:
            importlib.import_module(module)
        except ImportError:
            # Special handling for known package names vs import names
            # e.g. sklearn -> scikit-learn
            missing_packages.append(module)
        except Exception:
            # Other errors might indicate missing dependencies of dependencies
            pass

    print("\nPotentially Missing Packages (Import Name):")
    for p in missing_packages:
        print(f"- {p}")

if __name__ == "__main__":
    check_imports()
