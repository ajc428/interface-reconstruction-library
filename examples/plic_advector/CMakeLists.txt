set(CMAKE_CXX_COMPILER mpic++)

IRLAddExample_cpp(
  TARGET_NAME
  plic_advector
  HDRS
  "basic_mesh.h"
  "data.h"
  "deformation_3d.h"
  "translation_3d.h"
  "sheets.h"
  "reconstruction_types.h"
  "solver.h"
  "vof_advection.h"
  SRCS
  "main.cpp"
  "basic_mesh.cpp"
  "vof_advection.cpp"
  "solver.cpp"
  "reconstruction_types.cpp"
  "deformation_3d.cpp"
  "translation_3d.cpp"
  "sheets.cpp"
  OUTPUT_DIR
  "${IRL_EXAMPLES_OUTPUT_DIR}/plic_advector"
  COLLECTION
  "examples_plic_advector"
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(plic_advector PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET plic_advector PROPERTY CXX_STANDARD 17)