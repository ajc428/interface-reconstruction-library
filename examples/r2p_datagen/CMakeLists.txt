IRLAddExample_cpp(
  TARGET_NAME
    r2p_datagen
  HDRS
    "config.h"
    "scene.h"
    "stencil.h"
    "features.h"
  SRCS
    "main.cpp"
  OUTPUT_DIR
    "${IRL_EXAMPLES_OUTPUT_DIR}/r2p_datagen"
  COLLECTION
    "examples_r2p_datagen"
)
set_property(TARGET r2p_datagen PROPERTY CXX_STANDARD 17)

find_package(MPI REQUIRED)
target_link_libraries(r2p_datagen PRIVATE MPI::MPI_CXX)

IRLAddExample_cpp(
  TARGET_NAME
    r2p_eval
  SRCS
    "eval.cpp"
  OUTPUT_DIR
    "${IRL_EXAMPLES_OUTPUT_DIR}/r2p_datagen"
  COLLECTION
    "examples_r2p_datagen"
)
set_property(TARGET r2p_eval PROPERTY CXX_STANDARD 17)

# Needs libtorch, which the machine_learning example already requires.
IRLAddExample_cpp(
  TARGET_NAME
    r2p_export
  SRCS
    "export.cpp"
  OUTPUT_DIR
    "${IRL_EXAMPLES_OUTPUT_DIR}/r2p_datagen"
  COLLECTION
    "examples_r2p_datagen"
)
find_package(Torch REQUIRED)
target_link_libraries(r2p_export PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET r2p_export PROPERTY CXX_STANDARD 17)
