
from huggingface_hub import snapshot_download, login
import os


if __name__ == "__main__":
    login(
        new_session=False,  # Won’t request token if one is already saved on machine
        write_permission=False,  # Requires a token with write permission
        token=os.environ["HF_TOKEN"],
    )
    os.makedirs("networks/models", exist_ok=True)
    snapshot_download(
        repo_id="Ethics2025W/results",
        local_dir="results",
        repo_type="dataset",
    )
