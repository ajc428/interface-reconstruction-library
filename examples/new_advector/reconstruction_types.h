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
inline Data<int> feature_class;
inline Data<int> branch;
inline Data<int> snapped;   // slab planes (r2p_snap.h): 1 = thin-film snap, 2 = guard slab (the network wanted
                            // one plane where the thin-film guard holds), 3 = PCA slab (very thin film)
inline Data<double> unpinch;       // pinch prevention: rotation l toward the mean normal, 0 = none (r2p_nopinch.h)
inline Data<double> edge_sensor;   // film-edge sensor: distinct empty probe cells, -1 = not computed (r2p_edge_sensor.h)
inline Data<double> edge_topo;     // topological edge sensor: 1 edge, 0 not, 2 thick film, -1 not computed
                                   // (r2p_edge_topology.h)
inline Data<int> film_guard;       // thin-film guard (r2p_edge_topology.h): 1 sent to R2P-Net against the classifier,
                                   // 2 holds, the classifier already chose R2P-Net; pass 2 then keeps two planes
inline Data<int> one_plane_reason; // why an interface cell ended with one plane: 0 two planes (or not reconstructed),
                                   // 1 PLICNet by the classifier, 2 PLICNet without a classifier stencil (domain edge),
                                   // 3 the network predicted one plane, 4 pass-1 Newton clean-up dropped a plane,
                                   // 5 pinch prevention's re-solve dropped one, 6 pass-2 plane-count selection,
                                   // 7 pass-2 Newton clean-up dropped one
inline int r2p_dump_step = 0;     // time step written to the R2P_PLIC_DUMP file; set by the solver loop
inline Data<double> tip_sensor;   // film-tip sensor value: spread of the film centroids, cell^2 (r2p_tip_sensor.h)

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
