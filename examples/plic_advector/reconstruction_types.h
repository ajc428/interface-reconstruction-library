// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXAMPLES_PLIC_ADVECTOR_RECONSTRUCTION_TYPES_H_
#define EXAMPLES_PLIC_ADVECTOR_RECONSTRUCTION_TYPES_H_

#include <string>

#include "irl/planar_reconstruction/localized_separator_link.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/plic_advector/data.h"

void getReconstruction(
    const std::string& a_reconstruction_method,
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);

struct ML_PLIC {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction, const Data<IRL::Pt>& a_liquid_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface);
};

struct LVIRA3D {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const Data<IRL::Pt>& a_liquid_centroid,
                                const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface);
};

struct ELVIRA3D {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface);
};

void correctInterfacePlaneBorders(Data<IRL::PlanarSeparator>* a_interface);

#endif  // EXAMPLES_PARABOLOID_ADVECTOR_RECONSTRUCTION_TYPES_H_
