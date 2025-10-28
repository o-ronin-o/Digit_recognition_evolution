import os

# Define project structure
structure = {
    "project": {
        "data": [],
        "src": [
            "__init__.py",
            "utils.py",
            "plots.py",
        ],
        "notebooks": ["assignment_1.ipynb"],
        "results": [
        ],
        "files": ["env_mac.yml", "README.md"],
    }
}

# Helper function to create folders and files
def create_structure(base_path, structure):
    for folder, contents in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 Created folder: {folder_path}")

        for subfolder, subcontents in contents.items() if isinstance(contents, dict) else []:
            create_structure(folder_path, contents)

        if isinstance(contents, dict):
            for key, items in contents.items():
                if isinstance(items, list):
                    subpath = os.path.join(folder_path, key)
                    os.makedirs(subpath, exist_ok=True)
                    for item in items:
                        open(os.path.join(subpath, item), "a").close()
        elif isinstance(contents, list):
            for item in contents:
                open(os.path.join(folder_path, item), "a").close()

# Simpler builder for this structure
def setup_assignment():
    base = os.getcwd()
    root = os.path.join(base, "assignment_1")

    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    os.makedirs(os.path.join(root, "src"), exist_ok=True)
    os.makedirs(os.path.join(root, "notebooks"), exist_ok=True)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)

    files_to_create = [
        os.path.join(root, "env_mac.yml"),
        os.path.join(root, "README.md"),
        os.path.join(root, "notebooks", "assignment_1.ipynb"),
        os.path.join(root, "results", "classification_metrics.txt"),
        os.path.join(root, "results", "regression_metrics.txt"),
        os.path.join(root, "results", "knn_validation_plot.png"),
        os.path.join(root, "results", "regularization_plot.png"),
    ]

    src_files = [
        "__init__.py",
        "utils.py",
        "knn_manual.py",
        "regression_manual.py",
        "knn_sklearn.py",
        "regression_sklearn.py",
        "plots.py",
    ]

    for f in src_files:
        open(os.path.join(root, "src", f), "a").close()
    for f in files_to_create:
        open(f, "a").close()

    print("\n✅ Assignment structure successfully created at:", root)


if __name__ == "__main__":
    setup_assignment()
