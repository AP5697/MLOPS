from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize API
api = HfApi(token=HF_TOKEN)

# Space configuration
SPACE_ID = "Aishawarya/Bank-Customer-Churn"   # Replace with your actual username if different

# Create the Space if it doesn't exist
try:
    api.repo_info(
        repo_id=SPACE_ID,
        repo_type="space"
    )
    print(f"Space '{SPACE_ID}' already exists.")

except RepositoryNotFoundError:
    print(f"Space '{SPACE_ID}' not found. Creating it...")

    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="docker",     # Use "docker" because your deployment folder contains a Dockerfile
        private=False,
        token=HF_TOKEN,
    )

    print("Space created successfully!")

# Upload deployment files
api.upload_folder(
    folder_path="mlops/deployment",
    repo_id=SPACE_ID,
    repo_type="space",
    path_in_repo="",
)

print("Deployment files uploaded successfully!")
