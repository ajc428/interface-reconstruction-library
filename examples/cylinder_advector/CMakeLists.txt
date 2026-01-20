set(CMAKE_CXX_COMPILER mpic++)

IRLAddExample_cpp(
  TARGET_NAME
  cylinder_advector
  HDRS
  "basic_mesh.h"
  "data.h"
  "deformation_3d.h"
  "translation_3d.h"
  "reconstruction_types.h"
  "solver.h"
  "vof_advection.h"
  "vtk.h"
  SRCS
  "main.cpp"
  "basic_mesh.cpp"
  "vof_advection.cpp"
  "solver.cpp"
  "reconstruction_types.cpp"
  "deformation_3d.cpp"
  "translation_3d.cpp"
  "vtk.cpp"
  "trainer.h"
  "trainer.cpp"
  "neural_network.h"
  OUTPUT_DIR
  "${IRL_EXAMPLES_OUTPUT_DIR}/cylinder_advector"
  COLLECTION
  "examples_cylinder_advector"
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(cylinder_advector PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET cylinder_advector PROPERTY CXX_STANDARD 17)