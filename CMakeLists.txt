cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
project(demeter LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

include_directories(/homes/cjw19/dev/year3/individual/temp/samples/common/inc)

add_subdirectory(src)
