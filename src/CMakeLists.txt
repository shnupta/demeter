find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -G -g") # Debug

# set(DEMETERHEADERS
# 	common/common.cuh
# 	common/sample.cuh
# 	common/container.cuh
# 	product/product.cuh
# 	product/european.cuh
# 	product/binary_asian.cuh
# 	model/model.cuh
# 	model/black_scholes.cuh
# 	) 

# set_target_properties(libbonsai PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(demeter main.cu)
set_target_properties(demeter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(demeter -lcurand)
