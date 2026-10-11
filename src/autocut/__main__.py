"""Allow AutoCut to be run with: python -m autocut."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
