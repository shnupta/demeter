find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -G -g") # Debug

set(DEMETERHEADERS
	common.cuh
	) 

# set_target_properties(libbonsai PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(demeter main.cu ${DEMETERHEADERS})
set_target_properties(demeter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(demeter -lcurand)
