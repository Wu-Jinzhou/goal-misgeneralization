"""Keep nested test suites under the ``tests`` package namespace.

Without this marker, pytest can import ``tests/goalzendo`` as the top-level
``goalzendo`` package on Python 3.12, shadowing the package under ``src``.
"""
