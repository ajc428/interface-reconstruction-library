// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Fabien Evrard <fa.evrard@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXAMPLES_PARABOLOID_ADVECTOR_DEFORMATION_3D_H_
#define EXAMPLES_PARABOLOID_ADVECTOR_DEFORMATION_3D_H_

#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/plic_advector/basic_mesh.h"
#include "examples/plic_advector/data.h"

struct Deformation3D {
  static BasicMesh setMesh(void);

  static void initialize(Data<double>* a_U, Data<double>* a_V,
                         Data<double>* a_W,
                         Data<IRL::PlanarSeparator>* a_separators);

  static void setVelocity(const double a_time, Data<double>* a_U,
                          Data<double>* a_V, Data<double>* a_W);
};

#endif  // EXAMPLES_PARABOLOID_ADVECTOR_DEFORMATION_2D_H_
