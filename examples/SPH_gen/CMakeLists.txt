set(CMAKE_CXX_COMPILER mpic++)

IRLAddExample_cpp(
  TARGET_NAME
  SPH_gen
  HDRS
  "data_gen.h"
  SRCS
  "SPH_gen.cpp"
  OUTPUT_DIR
  "${IRL_EXAMPLES_OUTPUT_DIR}/SPH_gen"
  COLLECTION
  "examples_SPH_gen"
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(SPH_gen PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET SPH_gen PROPERTY CXX_STANDARD 17)