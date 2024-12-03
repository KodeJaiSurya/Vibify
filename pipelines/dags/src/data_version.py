from dvc.repo import Repo
from pathlib import Path
import os

def setup_dvc():
    """Initialize DVC in project root"""
    os.chdir("..")
    repo = Repo.init()
    repo.add_remote(
        name='storage',
        url=BUCKET_NAME,
        default=True
    )
    return repo

def add_data_to_dvc(data_path: str):
    repo = Repo()
    repo.add(data_path)
    repo.commit()

def main():
    setup_dvc()
    data_paths = ["data/raw", "data/preprocessed", "models"]
    
    for path in data_paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        add_data_to_dvc(path)

if __name__ == "__main__":
    main()