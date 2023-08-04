// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/plic_advector/reconstruction_types.h"

#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/interface_reconstruction_methods/elvira.h"
#include "irl/interface_reconstruction_methods/lvira_neighborhood.h"
#include "irl/interface_reconstruction_methods/lvira_optimization.h"
#include "irl/interface_reconstruction_methods/r2p_neighborhood.h"
#include "irl/interface_reconstruction_methods/r2p_optimization.h"
#include "irl/interface_reconstruction_methods/reconstruction_interface.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/localizer_link_from_localized_separator_link.h"

#include "examples/plic_advector/basic_mesh.h"
#include "examples/plic_advector/data.h"
#include "examples/plic_advector/vof_advection.h"
#include "irl/machine_learning_reconstruction/trainer.h"

void getReconstruction(
    const std::string& a_reconstruction_method,
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  if (a_reconstruction_method == "ELVIRA3D") {
    ELVIRA3D::getReconstruction(a_liquid_volume_fraction, a_dt, a_U, a_V, a_W,
                                a_interface);
  } else if (a_reconstruction_method == "ML_PLIC") {
    ML_PLIC::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_dt, a_U, a_V, a_W,
                                a_interface);
  } else {
    std::cout << "Unknown reconstruction method of : "
              << a_reconstruction_method << '\n';
    std::cout << "Value entries are: ELVIRA3D, ML_PLIC. \n";
    std::exit(-1);
  }
}

void ELVIRA3D::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                 const double a_dt, const Data<double>& a_U,
                                 const Data<double>& a_V,
                                 const Data<double>& a_W,
                                 Data<IRL::PlanarSeparator>* a_interface) {
  IRL::ELVIRANeighborhood neighborhood;
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  neighborhood.resize(27);
  IRL::RectangularCuboid cells[27];
  // Loop over cells in domain. Skip if cell is not mixed phase.

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(IRL::global_constants::ARBITRARILY_LARGE_DISTANCE,
                            a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }
        // Build surrounding stencil information for ELVIRA.
        for (int ii = i - 1; ii < i + 2; ++ii) {
          for (int jj = j - 1; jj < j + 2; ++jj) {
            for (int kk = k - 1; kk < k + 2; ++kk) {
              // Reversed order, bad for cache locality but thats okay..
              cells[(kk - k + 1) * 9 + (jj - j + 1) * 3 + (ii - i + 1)] =
                  IRL::RectangularCuboid::fromBoundingPts(
                      IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                      IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
              neighborhood.setMember(&cells[(kk - k + 1) * 9 + (jj - j + 1) * 3 + (ii - i + 1)],
                                    &a_liquid_volume_fraction(ii, jj, kk), ii - i,
                                    jj - j, kk - k);
            }
          }
        }
        // Now perform actual ELVIRA and obtain interface PlanarSeparator
        (*a_interface)(i, j, k) = reconstructionWithELVIRA3D(neighborhood);
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void ML_PLIC::getReconstruction(const Data<double>& a_liquid_volume_fraction, const Data<IRL::Pt>& a_liquid_centroid,
                                 const double a_dt, const Data<double>& a_U,
                                 const Data<double>& a_V,
                                 const Data<double>& a_W,
                                 Data<IRL::PlanarSeparator>* a_interface) {
  auto t = IRL::trainer(4);
  t.load_model("/home/andrew/Repositories/interface-reconstruction-library/examples/plic_advector/model.pt", 1);
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  // Loop over cells in domain. Skip if cell is not mixed phase.

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(IRL::global_constants::ARBITRARILY_LARGE_DISTANCE,
                            a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
        }
        else
        {
          // Build surrounding stencil information.
          auto n = IRL::Normal();
          Mesh local_mesh(3, 3, 3, 1);
          IRL::Pt lower_domain(-0.5 * local_mesh.getNx(), -0.5 * local_mesh.getNy(), -0.5 * local_mesh.getNz());
          IRL::Pt upper_domain(0.5 * local_mesh.getNx(), 0.5 * local_mesh.getNy(), 0.5 * local_mesh.getNz());
          local_mesh.setCellBoundaries(lower_domain, upper_domain);
          DataMesh<double> local_liquid_volume_fraction(local_mesh);
          DataMesh<IRL::Pt> local_liquid_centroid(local_mesh);
          for (int ii = i - 1; ii < i + 2; ++ii) {
            for (int jj = j - 1; jj < j + 2; ++jj) {
              for (int kk = k - 1; kk < k + 2; ++kk) {
                if (ii > mesh.imax() && jj > mesh.jmax() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                else if (ii < mesh.imin() && jj < mesh.jmin() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }
                else if (ii > mesh.imax() && jj > mesh.jmax() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }
                else if (ii > mesh.imax() && jj < mesh.jmin() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                else if (ii < mesh.imin() && jj > mesh.jmax() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                else if (ii > mesh.imax() && jj < mesh.jmin() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }
                else if (ii < mesh.imin() && jj > mesh.jmax() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }
                else if (ii < mesh.imin() && jj < mesh.jmax() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz(); 
                }


                else if (ii > mesh.imax() && jj > mesh.jmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (ii > mesh.imax() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), jj, mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                else if (jj > mesh.jmax() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, mesh.jmin(), mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                

                else if (ii > mesh.imax() && jj < mesh.jmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (ii > mesh.imax() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), jj, mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }
                else if (jj > mesh.jmax() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, mesh.jmin(), mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }


                else if (ii < mesh.imin() && jj < mesh.jmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (ii < mesh.imin() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), jj, mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }
                else if (jj < mesh.jmin() && kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, mesh.jmax(), mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }


                else if (ii < mesh.imin() && jj > mesh.jmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (ii < mesh.imin() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), jj, mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                else if (jj < mesh.jmin() && kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, mesh.jmax(), mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                
                
                else if (ii > mesh.imax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imin(), jj, kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imin(), jj, kk)[0] - mesh.xm(mesh.imin()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imin(), jj, kk)[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imin(), jj, kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (jj > mesh.jmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, mesh.jmin(), kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, mesh.jmin(), kk)[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (kk > mesh.kmax())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, jj, mesh.kmin());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, jj, mesh.kmin())[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz();
                }
                else if (ii < mesh.imin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(mesh.imax(), jj, kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(mesh.imax(), jj, kk)[0] - mesh.xm(mesh.imax()))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(mesh.imax(), jj, kk)[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(mesh.imax(), jj, kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (jj < mesh.jmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, mesh.jmax(), kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, mesh.jmax(), kk)[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz();
                }
                else if (kk < mesh.kmin())
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, jj, mesh.kmax());
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, jj, mesh.kmax())[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz();
                }


                else
                {
                  local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, jj, kk);
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[0] = (a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii))/mesh.dx();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[1] = (a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj))/mesh.dy();
                  local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1)[2] = (a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk))/mesh.dz();
                }
              }
            }
          }

          n = t.get_normal(local_liquid_volume_fraction, local_liquid_centroid);
          const IRL::Normal& n1 = n;
          const double d = a_liquid_volume_fraction(i,j,k);
          const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
          double distance = IRL::findDistanceOnePlane(cube, d, n1);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, distance));
      }
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void correctInterfacePlaneBorders(Data<IRL::PlanarSeparator>* a_interface) {
  const BasicMesh& mesh = (*a_interface).getMesh();
  // Fix distance to recreate volume fraction

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& plane : (*a_interface)(i, j, k)) {
          plane.distance() = plane.distance() - plane.normal()[0] * mesh.lx();
        }
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& plane : (*a_interface)(i, j, k)) {
          plane.distance() = plane.distance() + plane.normal()[0] * mesh.lx();
        }
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& plane : (*a_interface)(i, j, k)) {
          plane.distance() = plane.distance() - plane.normal()[1] * mesh.ly();
        }
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& plane : (*a_interface)(i, j, k)) {
          plane.distance() = plane.distance() + plane.normal()[1] * mesh.ly();
        }
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) {
        for (auto& plane : (*a_interface)(i, j, k)) {
          plane.distance() = plane.distance() - plane.normal()[2] * mesh.lz();
        }
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        for (auto& plane : (*a_interface)(i, j, k)) {
          plane.distance() = plane.distance() - plane.normal()[2] * mesh.lz();
        }
      }
    }
  }
}
