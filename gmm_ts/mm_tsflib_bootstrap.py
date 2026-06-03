"""Add MM-TSFlib to sys.path for models/ and layers/ imports."""
import os
import sys

MM_TSFLIB_PINNED_COMMIT = "e789ce78c9bafd8e3ba0d8850f9ad2becbe83548"


def bootstrap_mm_tsflib():
    mm_tsflib = os.environ.get("MM_TSFLIB_PATH")
    if not mm_tsflib:
        raise RuntimeError(
            "Set MM_TSFLIB_PATH to your MM-TSFlib clone "
            f"(tested at commit {MM_TSFLIB_PINNED_COMMIT})"
        )
    path = os.path.abspath(mm_tsflib)
    if not os.path.isdir(path):
        raise RuntimeError(f"MM_TSFLIB_PATH is not a directory: {path}")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
