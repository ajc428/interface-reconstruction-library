// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_INTERFACE_RECONSTRUCTION_METHODS_CYLINDER_RECONSTRUCTION_H_
#define IRL_INTERFACE_RECONSTRUCTION_METHODS_CYLINDER_RECONSTRUCTION_H_

#include <float.h>

#include <cassert>
#include <string>

#include "irl/cylinder_reconstruction/cylinder.h"
#include "irl/planar_reconstruction/planar_separator.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/cylinder_neighborhood.h"

namespace IRL {

class cylinder_reconstruction {

 public:
  /// \brief Default constructor.
  cylinder_reconstruction(void) = default;

  /// \brief Solve the for the reconstruction
  Cylinder solve(const cylinderNeighborhood* a_neighborhood_pointer, const int f);

  /// \brief Default destructor.
  ~cylinder_reconstruction(void) = default;

 private:
  /// \brief Solve the for the reconstruction.
  Cylinder solve(void);

  /// \brief Storage of the stencil information
  const cylinderNeighborhood* neighborhood_VF_m;
  int flip = 1;
};

class PrincipalCurve {
public:
    PrincipalCurve(const Eigen::MatrixXd& d, const Eigen::VectorXd& VFs, const double d_x);

    void fit(int max_iterations = 10, double tolerance = 1e-5);
    void fitPoly(IRL::Normal *direction, IRL::Pt *pt, IRL::Pt target);

    const Eigen::MatrixXd& getCurve() const;

private:
    int res = 3;
    double tol = 1e-8;
    double dx = 0;
    Eigen::MatrixXd data;
    Eigen::VectorXd VF;
    Eigen::MatrixXd curve;
    std::vector<std::vector<int>> projection_indices;
    Eigen::VectorXd lambda;

    void initializeWithPCA();
    
    void projectDataOntoCurve();
    
    void updateCurve();

    std::vector<double> solveCubic(double, double, double, double);
};

}  // namespace IRL

#endif // IRL_INTERFACE_RECONSTRUCTION_METHODS_CYLINDER_RECONSTRUCTION_H_
