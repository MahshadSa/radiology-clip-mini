import sys
import pathlib


def test_import_package():
    # Ensure src/ is on the path so we can import rclip without installation
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import rclip  # noqa: F401
