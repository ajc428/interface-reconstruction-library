// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Fabien Evrard <fa.evrard@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/new_advector/sheets.h"

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
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/localized_separator_link.h"

#include "examples/new_advector/data.h"
#include "examples/new_advector/reconstruction_types.h"
#include "examples/new_advector/solver.h"
#include "examples/new_advector/vof_advection.h"

constexpr int GC = 2;
constexpr IRL::Pt lower_domain(0.0, 0.0, 0.0);
constexpr IRL::Pt upper_domain(1.0, 1.0, 1.0);

BasicMesh Sheets::setMesh(const IRL::UnsignedIndex_t a_n) {
  BasicMesh mesh(a_n, a_n, a_n, GC);
  IRL::Pt my_lower_domain = lower_domain;
  IRL::Pt my_upper_domain = upper_domain;
  const double dx =
      (my_upper_domain[0] - my_lower_domain[0]) / static_cast<double>(a_n);
  mesh.setCellBoundaries(my_lower_domain, my_upper_domain);
  return mesh;
}

void Sheets::initialize(Data<double>* a_U, Data<double>* a_V,
                               Data<double>* a_W,
                               Data<IRL::PlanarSeparator>* a_separators) {
  Sheets::setVelocity(0.0, a_U, a_V, a_W);
  const BasicMesh& mesh = a_U->getMesh();
  constexpr int subdivisions = 1;
  IRL::PlanarSeparator temp_separator;
  // Loop over cells in domain. Skip if cell is not mixed phase.
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
        IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1));
        IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
            lower_cell_pt, upper_cell_pt);
        double dx = upper_cell_pt.x() - lower_cell_pt.x();
        double dy = upper_cell_pt.y() - lower_cell_pt.y();
        double dz = upper_cell_pt.z() - lower_cell_pt.z();
        double sc_dx = dx / static_cast<double>(subdivisions);
        double sc_dy = dy / static_cast<double>(subdivisions);
        double sc_dz = dz / static_cast<double>(subdivisions);
        IRL::Pt sc_lower;
        IRL::Pt sc_upper;

        // Create separators and localizers for sub-divided cell
        for (int ii = 0; ii < subdivisions; ++ii) {
          for (int jj = 0; jj < subdivisions; ++jj) {
            for (int kk = 0; kk < subdivisions; ++kk) {
              sc_lower[0] =
                  lower_cell_pt[0] + static_cast<double>(ii) * sc_dx;
              sc_lower[1] =
                  lower_cell_pt[1] + static_cast<double>(jj) * sc_dy;
              sc_lower[2] =
                  lower_cell_pt[2] + static_cast<double>(kk) * sc_dz;
              sc_upper[0] = sc_lower[0] + sc_dx;
              sc_upper[1] = sc_lower[1] + sc_dy;
              sc_upper[2] = sc_lower[2] + sc_dz;
              IRL::RectangularCuboid sub_cell =
                  IRL::RectangularCuboid::fromBoundingPts(sc_lower, sc_upper);
              IRL::Normal sub_cell_normal = IRL::Normal(0,0,1);


              double x_center = mesh.x(mesh.imax()/2);
              double x_left  = x_center + mesh.dx()/4.0;
              double x_right = x_center + 3.0*mesh.dx()/4.0;

              double cell_x_lo = mesh.x(i);
              double cell_x_hi = mesh.x(i+1);

              bool cell_fully_outside_film = (cell_x_hi <= x_left  || cell_x_lo >= x_right);

              if (cell_fully_outside_film) 
              {
                  (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
                     IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), -1));
                  // (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
                  //     IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), 1));

              } 
              else 
              {
                  (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(
                      IRL::Plane(IRL::Normal( 1.0, 0.0, 0.0),  x_right),
                      IRL::Plane(IRL::Normal(-1.0, -0.0, -0.0), -x_left),
                      1);
                  // (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(
                  //     IRL::Plane(IRL::Normal( 1.0, 0.0, 0.0),  x_left),
                  //     IRL::Plane(IRL::Normal(-1.0, 0.0, 0.0), -x_right),
                  //     -1);
                  // (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(
                  //     IRL::Plane(IRL::Normal( 1.0, 1.0, 1.0),  1),
                  //     IRL::Plane(IRL::Normal(-1.0, -1.0, -1.0), -0.98),
                  //     1);
              }

              //(*a_separators)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(
              //    IRL::Plane(IRL::Normal(1.0,-0.0,-0.0),(mesh.x(mesh.imaxo()/2)+3*mesh.dx()/4.0)),IRL::Plane(IRL::Normal(-1.0,-0.0,-0.0),-(mesh.x(mesh.imaxo()/2)+mesh.dx()/4.0)),1);
            }
          }
        }
      }
    }
  }
  //a_separators->updateBorder();
}

void Sheets::setVelocity(const double a_time, Data<double>* a_U,
                                Data<double>* a_V, Data<double>* a_W) {
  const BasicMesh& mesh = a_U->getMesh();
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        (*a_U)(i, j, k) = 0.1;
        (*a_V)(i, j, k) = 0;
        (*a_W)(i, j, k) = 0;
      }
    }
  }
}
