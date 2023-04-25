#List of files from this directory and its subdirectores.
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/neural_network.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/trainer.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/trainer.tpp)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/fractions.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/fractions.tpp)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/grad_functions.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/grad_functions.tpp)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/basic_mesh.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/basic_mesh.cpp)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/data.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/data_set.h)
target_sources(irl PRIVATE ${IRL_SOURCE_DIR}/machine_learning_reconstruction/data_set.tpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})