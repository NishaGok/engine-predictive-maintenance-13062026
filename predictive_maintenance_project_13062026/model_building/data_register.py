
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "NishaGok/engine-predictive-maintenance-13062026"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN1"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("Dataset repo already exists")

except RepositoryNotFoundError:
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    print("Dataset repo created")

api.upload_folder(
    folder_path="predictive_maintenance_project_13062026/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Dataset uploaded successfully")
