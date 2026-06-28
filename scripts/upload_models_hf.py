"""
One-time upload of trained models to Hugging Face Hub.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # only needed once, saves token to ~/.cache
    python scripts/upload_models_hf.py --repo YOUR_HF_USERNAME/acne-cv-models

What gets uploaded:
    models/model (42 dim)/*.pkl
    models/Evaluation_Summary(42dim).txt

The repo is created automatically (public by default; pass --private to hide).
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",    required=True, help="HF repo id, e.g. yourname/acne-cv-models")
    parser.add_argument("--private", action="store_true", help="Make repo private (requires HF Pro for large files)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Run:  pip install huggingface_hub")

    api = HfApi()

    print(f"Creating / verifying repo: {args.repo}")
    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)

    models_dir   = os.path.join(_ROOT, "models", "model (42 dim)")
    summary_file = os.path.join(_ROOT, "models", "Evaluation_Summary(42dim).txt")

    print(f"Uploading {models_dir}/ …")
    api.upload_folder(
        folder_path=models_dir,
        repo_id=args.repo,
        repo_type="model",
        path_in_repo="model_42dim",   # subfolder inside the HF repo
        commit_message="Upload 42-dim sklearn models",
    )

    print(f"Uploading summary …")
    api.upload_file(
        path_or_fileobj=summary_file,
        path_in_repo="Evaluation_Summary(42dim).txt",
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"\nDone. Set this env var on your deployed container:")
    print(f"  HF_MODEL_REPO={args.repo}")

if __name__ == "__main__":
    main()
