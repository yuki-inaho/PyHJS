cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME pyhjs)
project(${PROJ_NAME})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message("OPENMP IS ENABLED")
endif()


if(DEFINED PYTHON_INTERPRETER_VERSION_CALLING_SETUP_SCRIPT)
  set(PYTHON_INTERPRETER_VERSION ${PYTHON_INTERPRETER_VERSION_CALLING_SETUP_SCRIPT})
else()
  set(PYTHON_INTERPRETER_VERSION 3.8)
endif()

message("PYTHON_INTERPRETER_VERSION : " ${PYTHON_INTERPRETER_VERSION})
find_package(
  Python3 ${PYTHON_INTERPRETER_VERSION} EXACT
  COMPONENTS Interpreter Development NumPy
  REQUIRED)

message("PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS})
message("PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES})

add_subdirectory(extern/pybind11)
include_directories(include extern/pybind11/include)

pybind11_add_module(
  ${PROJ_NAME} ${PYTHON_INCLUDE_DIRS}
  src/skeleton.cpp
  src/anisotropic_diffusion.cpp
  src/bindings.cpp
  src/ndarray_converter.cpp)

pkg_check_modules(OpenCV REQUIRED opencv)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJ_NAME} ${OpenCV_LDFLAGS})

target_link_libraries(${PROJ_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJ_NAME} PUBLIC Python3::NumPy)
