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

  /// \brief Solve the system for the reconstruction, restarting
  /// the neighboring geoemtry
  Cylinder solve(const cylinderNeighborhood* a_neighborhood_pointer);

  /// \brief Default destructor.
  ~cylinder_reconstruction(void) = default;

 private:
  /// \brief Solve the system for the reconstruction.
  Cylinder solve(void);

  /// \brief Storage of the stencil information
  const cylinderNeighborhood* neighborhood_VF_m;
};

class PrincipalCurve {
public:
    PrincipalCurve(const Eigen::MatrixXd& d, const Eigen::VectorXd& VFs);

    void fit(int max_iterations = 100, double tolerance = 1e-5);
    void fitSpline(IRL::Normal *direction, IRL::Pt *pt, IRL::Pt target);

    const Eigen::MatrixXd& getCurve() const;

private:
    int res = 3;
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
};

}  // namespace IRL

#endif // IRL_INTERFACE_RECONSTRUCTION_METHODS_CYLINDER_RECONSTRUCTION_H_
