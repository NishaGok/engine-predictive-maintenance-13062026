
from huggingface_hub import HfApi
import os

api = HfApi(
    token=os.getenv("HF_TOKEN1")
)

api.upload_folder(

    folder_path=
    "predictive_maintenance_project_13062026/deployment",

    repo_id=
    "NishaGok/engine-predictive-maintenance-app-13062026",

    repo_type="space"

)

print("Deployment uploaded successfully")
