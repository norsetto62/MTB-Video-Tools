"""Allow AutoCut to be run with: python -m autocut."""

from . import __version__


def main() -> None:
    print(f"AutoCut {__version__}")


if __name__ == "__main__":
    main()