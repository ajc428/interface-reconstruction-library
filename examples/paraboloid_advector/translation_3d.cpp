// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Fabien Evrard <fa.evrard@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/paraboloid_advector/translation_3d.h"

#include <float.h>
#include <chrono>
#include <cmath>
#include <iostream>

#include "irl/distributions/k_means.h"
#include "irl/distributions/partition_by_normal_vector.h"
#include "irl/generic_cutting/cut_polygon.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/interface_reconstruction_methods/progressive_distance_solver_paraboloid.h"
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/localized_separator_link.h"

#include "examples/paraboloid_advector/data.h"
#include "examples/paraboloid_advector/reconstruction_types.h"
#include "examples/paraboloid_advector/solver.h"
#include "examples/paraboloid_advector/vof_advection.h"

constexpr int NX = 20;
constexpr int NY = 20;
constexpr int NZ = 20;
constexpr int GC = 3;
constexpr IRL::Pt lower_domain(0.0, 0.0, 0.0);
constexpr IRL::Pt upper_domain(1.0, 1.0, 1.0);

BasicMesh Translation3D::setMesh(void) {
  BasicMesh mesh(NX, NY, NZ, GC);
  IRL::Pt my_lower_domain = lower_domain;
  IRL::Pt my_upper_domain = upper_domain;
  mesh.setCellBoundaries(my_lower_domain, my_upper_domain);
  return mesh;
}

void Translation3D::initialize(Data<double>* a_U, Data<double>* a_V,
                               Data<double>* a_W,
                               Data<IRL::Paraboloid>* a_interface) {
  Translation3D::setVelocity(0.0, a_U, a_V, a_W);
  const BasicMesh& mesh = a_U->getMesh();
  const IRL::Pt sphere_center(0.5 + 0.0 * mesh.dx(), 0.5 + 0.0 * mesh.dx(),
                              0.5 + 0.0 * mesh.dx());
  const double sphere_radius = 0.25;

  // // Loop over cells in domain. Skip if cell is not mixed phase.
  // for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
  //   for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
  //     for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
  //       const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
  //       const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
  //                                   mesh.z(k + 1));
  //       const IRL::Pt mid_pt = 0.5 * (lower_cell_pt + upper_cell_pt);
  //       IRL::Pt disp = mid_pt - sphere_center;
  //       const auto mag = magnitude(disp);
  //       if (mag < sphere_radius - 2.0 * mesh.dx()) {
  //         (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
  //       } else if (mag > sphere_radius + 2.0 * mesh.dx()) {
  //         (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
  //       } else {
  //         auto sphere_normal = IRL::Normal::fromPt(disp);
  //         sphere_normal.normalize();
  //         (*a_interface)(i, j, k) =
  //             details::fromSphere(sphere_center, sphere_radius, sphere_normal);
  //       }
  //     }
  //   }
  // }

  // Loop over cells in domain. Skip if cell is not mixed phase.
  IRL::Pt ellipsoid_center = sphere_center;
  // double ellipsoid_rx = 1.5*sphere_radius;
  // double ellipsoid_ry = sphere_radius;
  // double ellipsoid_rz = 0.75*sphere_radius;
  double ellipsoid_rx = sphere_radius;
  double ellipsoid_ry = sphere_radius;
  double ellipsoid_rz = sphere_radius;
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
        const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                                        mesh.z(k + 1));
        const IRL::Pt mid_pt = 0.5 * (lower_cell_pt + upper_cell_pt);
        IRL::Pt disp = mid_pt - ellipsoid_center; // Use ellipsoid_center
        const double ellipsoid_val =
            pow(disp[0] / ellipsoid_rx, 2) + // Use disp[0] and ellipsoid_rx
            pow(disp[1] / ellipsoid_ry, 2) + // Use disp[1] and ellipsoid_ry
            pow(disp[2] / ellipsoid_rz, 2);  // Use disp[2] and ellipsoid_rz

        const double tolerance = 2; // Example tolerance value

        if (ellipsoid_val < 1.0 - tolerance) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
        } else if (ellipsoid_val > 1.0 + tolerance) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } else {
          // Calculate ellipsoid normal
          IRL::Normal ellipsoid_normal(2.0 * disp[0] / (ellipsoid_rx * ellipsoid_rx),
                                      2.0 * disp[1] / (ellipsoid_ry * ellipsoid_ry),
                                      2.0 * disp[2] / (ellipsoid_rz * ellipsoid_rz));
          ellipsoid_normal.normalize();

        // Calculate cell-based radius
        double disp_mag = magnitude(disp);
        double cell_radius = 0.0;
        if (disp_mag > 1e-9) { // Avoid division by zero if disp is very small
            cell_radius = disp_mag / std::sqrt(pow(disp[0]/ellipsoid_rx, 2) + pow(disp[1]/ellipsoid_ry, 2) + pow(disp[2]/ellipsoid_rz, 2));
        } else {
            cell_radius = ellipsoid_rx; // If at center, use ellipsoid_rx as default radius (or could use average radius)
        }


        // Use fromSphere with cell-based radius and ellipsoid_normal
        (*a_interface)(i, j, k) =
            details::fromSphere(ellipsoid_center, cell_radius, ellipsoid_normal);
        }
        //IRL::Normal normal = IRL::Normal(1/sqrt(3.0),1/sqrt(3.0),1/sqrt(3.0));
        // IRL::Normal normal = IRL::Normal(1.0,0.0,0.0);
        // normal.normalize();
        // double n2 = normal[0]/(sqrt(normal[1]*normal[1]+normal[0]*normal[0]));
        // double n1 = (-n2*normal[1])/normal[0];
        // IRL::Normal v1;
        // v1[0] = n1; v1[1] = n2; v1[2] = 0;
        // IRL::Normal b = IRL::crossProduct(normal,v1);
        // b.normalize();
        // IRL::Normal a = IRL::crossProduct(b,normal);
        // a.normalize();
        // IRL::ReferenceFrame frame = IRL::ReferenceFrame(a, b, normal);
        // IRL::Pt datum = IRL::Pt(mesh.xm(9), mesh.ym(9), mesh.zm(9));
        // IRL::Paraboloid p = IRL::Paraboloid(datum,frame,3,0.25);
        // auto cell = IRL::RectangularCuboid::fromBoundingPts(
        //   IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
        //   IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        // auto moments = IRL::getVolumeMoments<IRL::VolumeMoments, IRL::HalfEdgeCutting>(cell, p);
        // if (moments.volume() <= IRL::global_constants::VF_LOW)
        // {
        //   (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        // }
        // else if (moments.volume() >= IRL::global_constants::VF_HIGH)
        // {
        //   (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
        // }
        // else
        // {
        //   (*a_interface)(i, j, k) = p;
        // }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void Translation3D::setVelocity(const double a_time, Data<double>* a_U,
                                Data<double>* a_V, Data<double>* a_W) {
  const BasicMesh& mesh = a_U->getMesh();
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        (*a_U)(i, j, k) = 1.0;
        (*a_V)(i, j, k) = 1.0 / 1.5;
        (*a_W)(i, j, k) = 1.0 / 3.0;
      }
    }
  }
}
