// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2019 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXAMPLES_NEW_ADVECTOR_RECONSTRUCTION_TYPES_H_
#define EXAMPLES_NEW_ADVECTOR_RECONSTRUCTION_TYPES_H_

#include <string>

#include "irl/planar_reconstruction/localized_separator_link.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/new_advector/data.h"

inline Data<int> recon_method;
inline Data<int> num_planes;

void getReconstruction(
    const std::string& a_reconstruction_method,
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);

struct ELVIRA2D {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,
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

struct LVIRA2D {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const Data<IRL::Pt>& a_liquid_centroid,
                                const Data<IRL::Pt>& a_gas_centroid,
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

struct PLICNET {
  static void getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const Data<IRL::Pt>& a_liquid_centroid,
                                const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface);
};

struct MOF2D {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct MOF3D {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct AdvectedNormals {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct AdvectedNormals3D {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct R2P2D {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct R2P3D {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct R2P3D_Hybrid {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct R2P3D_Net {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

void correctInterfacePlaneBorders(Data<IRL::PlanarSeparator>* a_interface);

#endif  // EXAMPLES_NEW_ADVECTOR_RECONSTRUCTION_TYPES_H_
