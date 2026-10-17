"""Entry point for ``python -m aar``; defers to the same CLI as the ``aar`` script."""

from __future__ import annotations

from aar.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
