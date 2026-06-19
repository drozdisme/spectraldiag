cmake_minimum_required(VERSION 3.18)
project(spectraldiag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /EHsc)
else()
    add_compile_options(-O3 -ffast-math)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

Python_add_library(_core MODULE src/pymodule.cpp)
target_include_directories(_core PRIVATE src)

install(TARGETS _core DESTINATION spectraldiag)
