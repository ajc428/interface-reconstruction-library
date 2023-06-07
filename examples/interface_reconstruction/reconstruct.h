#include <string>

#include "irl/paraboloid_reconstruction/paraboloid.h"
#include "irl/planar_reconstruction/planar_separator.h"
#include "irl/machine_learning_reconstruction/basic_mesh.h"

#include "irl/machine_learning_reconstruction/data.h"

void getReconstruction(const std::string& a_reconstruction_method, const Data<double>& a_liquid_volume_fraction, Data<IRL::Paraboloid>* a_interface);

struct PLIC 
{
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction, Data<IRL::Paraboloid>* a_interface);
};

struct Jibben 
{
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction, Data<IRL::Paraboloid>* a_interface);
};

struct Centroid 
{
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction, Data<IRL::Paraboloid>* a_interface);
};

void correctInterfacePlaneBorders(Data<IRL::Paraboloid>* a_interface);

