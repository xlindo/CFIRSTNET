cmake_minimum_required(VERSION 3.0.0)
project(ICCAD VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O3 -std=c++17")

find_package(PythonLibs REQUIRED)
include_directories( ${PYTHON_INCLUDE_DIRS} )

find_package(Boost COMPONENTS python numpy)
include_directories (${Boost_INCLUDE_DIRS})

add_library(data SHARED main.cpp)
target_link_libraries(
    data
    ${PYTHON_LIBRARIES}
    ${Boost_LIBRARIES}
    -fPIC -shared
)