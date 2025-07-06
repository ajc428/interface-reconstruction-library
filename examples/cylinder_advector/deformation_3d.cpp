// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Fabien Evrard <fa.evrard@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/cylinder_advector/deformation_3d.h"

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

#include "examples/cylinder_advector/data.h"
#include "examples/cylinder_advector/reconstruction_types.h"
#include "examples/cylinder_advector/solver.h"
#include "examples/cylinder_advector/vof_advection.h"

constexpr int NX = 20;
constexpr int NY = 20;
constexpr int NZ = 20;
constexpr int GC = 3;
constexpr IRL::Pt lower_domain(0.0, 0.0, 0.0);
constexpr IRL::Pt upper_domain(1, 1, 1);

BasicMesh Deformation3D::setMesh(void) {
  BasicMesh mesh(NX, NY, NZ, GC);
  IRL::Pt my_lower_domain = lower_domain;
  IRL::Pt my_upper_domain = upper_domain;
  mesh.setCellBoundaries(my_lower_domain, my_upper_domain);
  return mesh;
}

void Deformation3D::initialize(Data<double>* a_U, Data<double>* a_V,
                               Data<double>* a_W,
                               Data<IRL::Paraboloid>* a_interface) {
  Deformation3D::setVelocity(0.0, a_U, a_V, a_W);
  const BasicMesh& mesh = a_U->getMesh();
  const IRL::Pt sphere_center(0.35, 0.35, 0.35);
  const double sphere_radius = 0.15;

  // Loop over cells in domain. Skip if cell is not mixed phase.
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
        const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                                    mesh.z(k + 1));
        const IRL::Pt mid_pt = 0.5 * (lower_cell_pt + upper_cell_pt);
        IRL::Pt disp = mid_pt - sphere_center;
        const auto mag = magnitude(disp);
        if (mag < sphere_radius - 2.0 * mesh.dx()) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
        } else if (mag > sphere_radius + 2.0 * mesh.dx()) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } else {
          auto sphere_normal = IRL::Normal::fromPt(disp);
          sphere_normal.normalize();
          (*a_interface)(i, j, k) =
              details::fromSphere(sphere_center, sphere_radius, sphere_normal);
        }
      }
    }
  }
  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}





void Deformation3D::initialize(Data<double>* a_U, Data<double>* a_V,
                               Data<double>* a_W,
                               Data<IRL::Cylinder>* a_interface) {
  Deformation3D::setVelocity2(0.0, a_U, a_V, a_W);
  const BasicMesh& mesh = a_U->getMesh();
  auto cell = IRL::RectangularCuboid::fromBoundingPts(
  IRL::Pt(mesh.x(9), mesh.y(9), mesh.z(9)),
  IRL::Pt(mesh.x(9 + 1), mesh.y(9 + 1), mesh.z(9 + 1)));
  IRL::HalfEdgePolyhedronQuadratic<IRL::Pt> half_edge;
  cell.setHalfEdgeVersion(&half_edge);
  auto seg_half_edge = half_edge.generateSegmentedPolyhedron();
  std::ofstream myfile;
  myfile.open("cell.vtu");
  myfile << seg_half_edge;
  myfile.close();
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        IRL::Normal normal = IRL::Normal(1.0,0.0,0.0);
        normal.normalize();
        double n2 = normal[0]/(sqrt(normal[1]*normal[1]+normal[0]*normal[0]));
        double n1 = (-n2*normal[1])/normal[0];
        IRL::Normal v1;
        v1[0] = n1; v1[1] = n2; v1[2] = 0;
        IRL::Normal b = IRL::crossProduct(normal,v1);
        b.normalize();
        IRL::Normal a = IRL::crossProduct(b,normal);
        a.normalize();
        IRL::ReferenceFrame frame = IRL::ReferenceFrame(normal, a, b);
        IRL::Pt datum = IRL::Pt(mesh.xm(9), mesh.ym(9), mesh.zm(9));
        IRL::Cylinder p = IRL::Cylinder(datum,frame,1,0.000125);
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
          IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        auto moments = IRL::getVolumeMoments<IRL::VolumeMoments, IRL::HalfEdgeCutting>(cell, p);
        if (moments.volume() <= IRL::global_constants::VF_LOW)
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysBelow();
        }
        else if (moments.volume() >= IRL::global_constants::VF_HIGH)
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysAbove();
        }
        else
        {
          (*a_interface)(i, j, k) = p;
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void Deformation3D::setVelocity(const double a_time, Data<double>* a_U,
                                Data<double>* a_V, Data<double>* a_W) {
  const BasicMesh& mesh = a_U->getMesh();
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        (*a_U)(i, j, k) = 2.0 * std::pow(sin(M_PI * mesh.xm(i)), 2) *
                          sin(2.0 * M_PI * mesh.ym(j)) *
                          sin(2.0 * M_PI * mesh.zm(k)) *
                          cos(M_PI * (a_time) / 3.0);
        (*a_V)(i, j, k) = -std::pow(sin(M_PI * mesh.ym(j)), 2) *
                          sin(2.0 * M_PI * mesh.xm(i)) *
                          sin(2.0 * M_PI * mesh.zm(k)) *
                          cos(M_PI * (a_time) / 3.0);
        (*a_W)(i, j, k) = -std::pow(sin(M_PI * mesh.zm(k)), 2) *
                          sin(2.0 * M_PI * mesh.xm(i)) *
                          sin(2.0 * M_PI * mesh.ym(j)) *
                          cos(M_PI * (a_time) / 3.0);
      }
    }
  }
}

void Deformation3D::setVelocity2(const double a_time, Data<double>* a_U,
                                Data<double>* a_V, Data<double>* a_W) {
  const BasicMesh& mesh = a_U->getMesh();
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        // (*a_U)(i, j, k) = 2.0 * std::pow(sin(M_PI * mesh.xm(i)), 2) *
        //                   sin(2.0 * M_PI * mesh.ym(j)) *
        //                   sin(2.0 * M_PI * mesh.zm(k)) *
        //                   cos(M_PI * (a_time) / 3.0);
        // (*a_V)(i, j, k) = -std::pow(sin(M_PI * mesh.ym(j)), 2) *
        //                   sin(2.0 * M_PI * mesh.xm(i)) *
        //                   sin(2.0 * M_PI * mesh.zm(k)) *
        //                   cos(M_PI * (a_time) / 3.0);
        // (*a_W)(i, j, k) = -std::pow(sin(M_PI * mesh.zm(k)), 2) *
        //                   sin(2.0 * M_PI * mesh.xm(i)) *
        //                   sin(2.0 * M_PI * mesh.ym(j)) *
        //                   cos(M_PI * (a_time) / 3.0);
        (*a_U)(i, j, k) = 0.0;
        (*a_V)(i, j, k) = 2*sin(2 * M_PI * mesh.xm(i));
        (*a_W)(i, j, k) = 0.0;
      }
    }
  }
}
