import os

# List of directories to create
folders = [
    "src/config",
    "src/data",
    "src/preprocess",
    "src/train",
    "src/evaluate",
    "src/predict",
    "src/utils",
    "config",
    "data/raw",
    "data/processed",
    "notebooks",
    "logs",
    "experiments",
]

# List of empty files to initialize
files = [
    "main.py",
    "README.md",
    "requirements.txt",
    "setup.py",
    "config/config.yaml",
    "src/__init__.py",
    "src/config/config_loader.py",
    "src/data/data_loader.py",
    "src/preprocess/preprocess.py",
    "src/train/model_trainer.py",
    "src/evaluate/model_evaluator.py",
    "src/predict/predictor.py",
    "src/utils/logger.py",
    "src/utils/exception.py",
]

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "__init__.py"), "w") as f:
            pass  

    for file in files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                f.write("")  

    print("Project structure created successfully.")

if __name__ == "__main__":
    create_structure()
