import sys
from setuptools import setup, Extension

if sys.platform == "win32":
    # /utf-8: treat source and execution charset as UTF-8 so \u escapes
    # in string literals emit UTF-8 bytes (not the system codepage).
    extra_compile_args = ["/O2", "/std:c++17", "/EHsc", "/utf-8"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-std=c++17", "-ffast-math"]
    extra_link_args = []

ext = Extension(
    "spectraldiag._core",
    sources=["src/pymodule.cpp"],
    include_dirs=["src"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(ext_modules=[ext])
