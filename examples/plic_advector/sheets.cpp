// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Fabien Evrard <fa.evrard@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/plic_advector/sheets.h"

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

#include "examples/plic_advector/data.h"
#include "examples/plic_advector/reconstruction_types.h"
#include "examples/plic_advector/solver.h"
#include "examples/plic_advector/vof_advection.h"

constexpr int NX = 10;
constexpr int NY = 10;
constexpr int NZ = 10;
constexpr int GC = 2;
constexpr IRL::Pt lower_domain(0.0, 0.0, 0.0);
constexpr IRL::Pt upper_domain(1.0, 1.0, 1.0);

BasicMesh Sheets::setMesh(void) {
  BasicMesh mesh(NX, NY, NZ, GC);
  IRL::Pt my_lower_domain = lower_domain;
  IRL::Pt my_upper_domain = upper_domain;
  mesh.setCellBoundaries(my_lower_domain, my_upper_domain);
  return mesh;
}

void Sheets::initialize(Data<double>* a_U, Data<double>* a_V,
                               Data<double>* a_W,
                               Data<IRL::PlanarSeparator>* a_separators) {
  Sheets::setVelocity(0.0, a_U, a_V, a_W);
  const BasicMesh& mesh = a_U->getMesh();
  //const IRL::Pt sphere_center(0.5 + 0.0 * mesh.dx(), 0.5 + 0.0 * mesh.dx(),
  //                            0.5 + 0.0 * mesh.dx());
  const double sheet_width = 0.25;
  constexpr int subdivisions = 1;
  IRL::PlanarSeparator temp_separator;
  IRL::ListedVolumeMoments<IRL::VolumeMoments> listed_volume_moments;
  // Loop over cells in domain. Skip if cell is not mixed phase.
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        const IRL::Pt sheet_center(mesh.x(i), mesh.y(j),
                              0.5 + 0.0 * mesh.dx());
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
        const IRL::Pt mid_pt = 0.5 * (lower_cell_pt + upper_cell_pt);
        IRL::Pt disp = mid_pt - sheet_center;
        const auto mag = magnitude(disp);
        //if (mesh.z(k) < 0.2) {
        //  (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
        //      IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), -10000000.0));
        //} else if (mesh.z(k) > 0.3) {
         // (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
        //      IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), -10000000.0));
        //} else {
          listed_volume_moments.clear();
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
                IRL::Normal sub_cell_normal = IRL::Normal(0,0,1);//::fromPtNormalized(
                    //(sub_cell.calculateCentroid() - sheet_center));
                (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(
                    IRL::Plane(IRL::Normal(-1.0,-1.0,-1.0),-0.92),IRL::Plane(IRL::Normal(1.0,1.0,1.0),1),1);
                //IRL::Polygon interface_poly =
                //    IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                //        sub_cell, temp_separator, temp_separator[0]);
                //auto moments =
                    //IRL::getVolumeMoments<IRL::VolumeMoments>(sub_cell,temp_separator);
                //listed_volume_moments += moments;
              //}
            }
          }
          // IRL::VolumeMomentsAndNormal mean_moments;
          // for (const auto& element : listed_volume_moments) {
          //   mean_moments += element;
          // }
          // if (mean_moments.volumeMoments().volume() == 0.0) {
          //   (*a_separators)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
          //       IRL::Plane(IRL::Normal(0.0, 0.0, 0.0),
          //                  std::copysign(1.0, sheet_width -
          //                                         IRL::magnitude(
          //                                             cell.calculateCentroid() -
          //                                             sheet_center))));
          // } else {
          //   mean_moments.normalizeByVolume();
          //   mean_moments.normal().normalize();
        }
          //       IRL::Plane(mean_moments.normal(),
          //                  mean_moments.normal() *
          //                      mean_moments.volumeMoments().centroid()));
          // }
        //}
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
        //if (mesh.zm(k) > 0.25)
        {
          (*a_U)(i, j, k) = 0;
          (*a_V)(i, j, k) = 0;
          (*a_W)(i, j, k) = 0;
        }
        //else
        {
          //(*a_U)(i, j, k) = 0;
          //(*a_V)(i, j, k) = 0;
          //(*a_W)(i, j, k) = 0.1;
        }
      }
    }
  }
}
