import sys
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import HfApi

parser = ArgumentParser()
parser.add_argument("repo_id")
parser.add_argument("folder", type=Path)

args = parser.parse_args()

if not args.folder.is_dir():
    print(f'{args.folder} is not a directory!', file=sys.stderr)
    sys.exit(1)


api = HfApi()

api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id
)

