"""Internal entry point for the supervised status API."""

import argparse

import uvicorn

from .history import History
from .profile import load_profile
from .status import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--database", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    app = create_app(profile, History(args.database))
    uvicorn.run(app, host=profile["status"]["host"], port=profile["status"]["port"], log_level="warning")


if __name__ == "__main__":
    main()
