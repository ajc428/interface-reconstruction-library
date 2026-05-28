#List of files from this directory and its subdirectores.
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/neural_network.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/trainer.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/trainer.cpp)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/stencil.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/moments_gen.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/moments_gen.cpp)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/data_set.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/data_set.cpp)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/data_gen.h)

find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_PREFIX_PATH "/home/andrew/Repositories/libtorch/share/cmake/Torch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})