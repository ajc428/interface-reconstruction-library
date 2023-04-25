IRLAddExample_cpp(
  TARGET_NAME
    machine_learning
  SRCS
    "test.cpp"
  OUTPUT_DIR
    # "${IRL_EXAMPLES_OUTPUT_DIR}/machine_learning"
    "/home/andrew/Repositories/interface-reconstruction-library/examples/machine_learning"
  COLLECTION
    "examples_machine_learning"
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(machine_learning PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET machine_learning PROPERTY CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(machine_learning PRIVATE MPI::MPI_CXX)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(machine_learning PRIVATE ${PYTHON_LIBRARIES})