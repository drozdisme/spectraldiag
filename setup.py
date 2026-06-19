import sys
from setuptools import setup, Extension

extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17", "/EHsc"]
else:
    extra_compile_args = ["-O3", "-std=c++17", "-ffast-math"]

ext = Extension(
    "spectraldiag._core",
    sources=["src/pymodule.cpp"],
    include_dirs=["src"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(ext_modules=[ext])
