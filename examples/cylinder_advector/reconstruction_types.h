// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXAMPLES_CYLINDER_ADVECTOR_RECONSTRUCTION_TYPES_H_
#define EXAMPLES_CYLINDER_ADVECTOR_RECONSTRUCTION_TYPES_H_

#include <string>

#include "irl/paraboloid_reconstruction/paraboloid.h"
#include "irl/cylinder_reconstruction/cylinder.h"
#include "irl/planar_reconstruction/planar_separator.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/cylinder_reconstruction.h"
#include "irl/interface_reconstruction_methods/cylinder_neighborhood.h"

#include "examples/cylinder_advector/data.h"

void getReconstruction(const std::string& a_reconstruction_method,
                       const Data<double>& a_liquid_volume_fraction,
                       const Data<IRL::Pt>& a_liquid_centroid,
                       const Data<IRL::Pt>& a_gas_centroid,
                       const Data<IRL::LocalizedParaboloidLink<double>>&
                           a_localized_paraboloid_link,
                       const double a_dt, const Data<double>& a_U,
                       const Data<double>& a_V, const Data<double>& a_W,
                       Data<IRL::Paraboloid>* a_interface);

void getReconstruction(const std::string& a_reconstruction_method,
                       const Data<double>& a_liquid_volume_fraction,
                       const Data<IRL::Pt>& a_liquid_centroid,
                       const Data<IRL::Pt>& a_gas_centroid,
                       const Data<IRL::LocalizedCylinderLink<double>>&
                           a_localized_cylinder_link,
                       const double a_dt, const Data<double>& a_U,
                       const Data<double>& a_V, const Data<double>& a_W,
                       Data<IRL::Cylinder>* a_interface);    

struct PLIC_NET {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,const Data<IRL::Pt>& a_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface);
};                   

struct PLIC {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::Paraboloid>* a_interface);
};

struct Jibben {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,const Data<IRL::Pt>& a_liquid_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::Paraboloid>* a_interface);
};

struct Cylinder_PCA {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,const Data<IRL::Pt>& a_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::Cylinder>* a_interface);
};

struct Cylinder_Spline {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,const Data<IRL::Pt>& a_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::Cylinder>* a_interface);
};

struct Cylinder_Curve_Global {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,const Data<IRL::Pt>& a_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::Cylinder>* a_interface);
};

struct Cylinder_Curve_Local {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,const Data<IRL::Pt>& a_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::Cylinder>* a_interface);
};

void correctInterfacePlaneBorders(Data<IRL::Paraboloid>* a_interface);

void correctInterfacePlaneBorders(Data<IRL::Cylinder>* a_interface);

void correctInterfacePlaneBorders(Data<IRL::PlanarSeparator>* a_interface);

void load();

namespace details {
inline IRL::Paraboloid fromSphere(const IRL::Pt& a_center,
                                  const double a_radius,
                                  const IRL::Normal& a_normal) {
  const double curvature = 1.0 / a_radius;
  IRL::ReferenceFrame frame;
  int largest_dir = 0;
  if (std::fabs(a_normal[largest_dir]) < std::fabs(a_normal[1]))
    largest_dir = 1;
  if (std::fabs(a_normal[largest_dir]) < std::fabs(a_normal[2]))
    largest_dir = 2;
  if (largest_dir == 0)
    frame[0] = IRL::crossProduct(a_normal, IRL::Normal(0.0, 1.0, 0.0));
  else if (largest_dir == 1)
    frame[0] = IRL::crossProduct(a_normal, IRL::Normal(0.0, 0.0, 1.0));
  else
    frame[0] = IRL::crossProduct(a_normal, IRL::Normal(1.0, 0.0, 0.0));
  frame[0].normalize();
  frame[1] = crossProduct(a_normal, frame[0]);
  frame[2] = a_normal;

  return IRL::Paraboloid(a_center + a_radius * a_normal, frame, 0.5 * curvature,
                         0.5 * curvature);
}
}

class PrincipalCurve {
public:
    PrincipalCurve(const Eigen::MatrixXd& d, const Eigen::VectorXd& VFs, const double d_x);

    void fit(int max_iterations = 10, double tolerance = 1e-5);
    void fitPoly(IRL::Normal *direction, IRL::Pt *pt, IRL::Pt target);

    const Eigen::MatrixXd& getCurve() const;

private:
    int res = 3;
    double dx = 0;
    Eigen::MatrixXd data;
    Eigen::VectorXd VF;
    Eigen::MatrixXd curve;
    Eigen::VectorXi projection_indices;
    Eigen::VectorXd lambda;

    void initializeWithPCA();
    
    void projectDataOntoCurve();
    
    void updateCurve();

    void smoothCurve(int window_size = 3);

    void orderCurvePoints();

    std::vector<double> solveCubic(double, double, double, double);
};

#endif  // EXAMPLES_CYLINDER_ADVECTOR_RECONSTRUCTION_TYPES_H_
