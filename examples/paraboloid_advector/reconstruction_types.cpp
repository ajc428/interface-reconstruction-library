// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/paraboloid_advector/reconstruction_types.h"

#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/interface_reconstruction_methods/constrained_optimization_behavior.h"
#include "irl/interface_reconstruction_methods/elvira_neighborhood.h"
#include "irl/interface_reconstruction_methods/plvira_neighborhood.h"
#include "irl/interface_reconstruction_methods/progressive_distance_solver_paraboloid.h"
#include "irl/interface_reconstruction_methods/reconstruction_interface.h"
#include "irl/moments/volume_moments_with_gradient.h"
#include "irl/moments/volume_with_gradient.h"
#include "irl/optimization/constrained_levenberg_marquardt.h"
#include "irl/paraboloid_reconstruction/gradient_paraboloid.h"
#include "irl/paraboloid_reconstruction/hessian_paraboloid.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include <Eigen/Dense>
#include "examples/paraboloid_advector/basic_mesh.h"
#include "examples/paraboloid_advector/data.h"
#include "examples/paraboloid_advector/vof_advection.h"
#include "irl/machine_learning_reconstruction/trainer.h"

auto t = IRL::trainer(1);
auto t_axis = IRL::trainer(0);
auto t_coeff = IRL::trainer(-1);
auto t_origin = IRL::trainer(-2);

void load()
{
  t.load_model("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model.pt");
  t_axis.load_model("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model_axis.pt");
  t_coeff.load_model("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model_coeff.pt");
  t_origin.load_model("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model_origin.pt");
};

void getReconstruction(const std::string& a_reconstruction_method,
                       const Data<double>& a_liquid_volume_fraction,
                       const Data<IRL::Pt>& a_liquid_centroid,
                       const Data<IRL::Pt>& a_gas_centroid,
                       const Data<IRL::LocalizedParaboloidLink<double>>&
                           a_localized_paraboloid_link,
                       const double a_dt, const Data<double>& a_U,
                       const Data<double>& a_V, const Data<double>& a_W,
                       Data<IRL::Paraboloid>* a_interface) {
  if (a_reconstruction_method == "Jibben") {
    Jibben::getReconstruction(a_liquid_volume_fraction, a_dt, a_U, a_V, a_W,
                              a_interface);
  } else if (a_reconstruction_method == "CentroidFit") {
    Centroid::getReconstruction(a_liquid_volume_fraction, a_dt, a_U, a_V, a_W,
                                a_interface);
  } else if (a_reconstruction_method == "PLIC") {
    PLIC::getReconstruction(a_liquid_volume_fraction, a_dt, a_U, a_V, a_W,
                            a_interface);
  } else if (a_reconstruction_method == "ML") {
    ML::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_interface);
  } else if (a_reconstruction_method == "ML_norm") {
    ML_norm::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_interface);
  }  else if (a_reconstruction_method == "ML_QUAD") {
    ML_QUAD::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid, a_dt, a_U, a_V, a_W,
                                a_interface);
  } else {
    std::cout << "Unknown reconstruction method of : "
              << a_reconstruction_method << '\n';
    std::cout << "Valid entries are: PLIC, CentroidFit, Jibben, ML, ML_norm. \n";
    std::exit(-1);
  }
}

// Wendland radial basis function
// Wendland, H. (1995). Piecewise polynomial, positive definite and
// compactly supported radial functions of minimal degree. Advances in
// Computational Mathematics, 4(1), 389â€“396.
double wgauss(const double d, const double h) {
  if (d >= h) {
    return 0.0;
  } else {
    return (1.0 + 4.0 * d / h) * std::pow(1.0 - d / h, 4.0);
  }
}

void updateReconstructionELVIRA(
    const Data<double>& a_liquid_volume_fraction,
    Data<IRL::PlanarSeparator>* a_liquid_gas_interface) {
  IRL::ELVIRANeighborhood neighborhood;
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  neighborhood.resize(27);
  IRL::RectangularCuboid cells[27];
  // Loop over cells in domain. Skip if cell is not mixed phase.
  for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) >
                IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_liquid_gas_interface)(i, j, k) =
              IRL::PlanarSeparator::fromOnePlane(
                  IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }
        // Build surrounding stencil information for ELVIRA.
        for (int kk = k - 1; kk < k + 2; ++kk) {
          for (int jj = j - 1; jj < j + 2; ++jj) {
            for (int ii = i - 1; ii < i + 2; ++ii) {
              // Reversed order, bad for cache locality but thats okay..
              cells[(kk - k + 1) * 9 + (jj - j + 1) * 3 + (ii - i + 1)] =
                  IRL::RectangularCuboid::fromBoundingPts(
                      IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                      IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
              neighborhood.setMember(
                  &cells[(kk - k + 1) * 9 + (jj - j + 1) * 3 + (ii - i + 1)],
                  &a_liquid_volume_fraction(ii, jj, kk), ii - i, jj - j,
                  kk - k);
            }
          }
        }
        // Now perform actual ELVIRA and obtain interface PlanarSeparator
        (*a_liquid_gas_interface)(i, j, k) =
            reconstructionWithELVIRA3D(neighborhood);
      }
    }
  }
  // Update border with simple ghost-cell fill and correct distances for
  // assumed periodic boundary
  a_liquid_gas_interface->updateBorder();
  // correctInterfacePlaneBorders(a_liquid_gas_interface);
}

// Reconstruction with LVIRA - use input PlanarSeparator as initial guess
void updateReconstructionLVIRA(
    const Data<double>& a_liquid_volume_fraction, const int a_nneigh,
    Data<IRL::PlanarSeparator>* a_liquid_gas_interface) {
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();

  IRL::LVIRANeighborhood<IRL::RectangularCuboid> neighborhood;
  std::vector<IRL::RectangularCuboid> cells;
  // std::vector<double> weights; // maybe later

  const int grid_size =
      (a_nneigh * 2 + 1) * (a_nneigh * 2 + 1) * (a_nneigh * 2 + 1);
  neighborhood.resize(grid_size);
  cells.resize(grid_size);

  // Loop over cells in domain. Skip if cell is not mixed phase.
  for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) >
                IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_liquid_gas_interface)(i, j, k) =
              IRL::PlanarSeparator::fromOnePlane(
                  IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }

        // Build surrounding stencil information for LVIRA.
        IRL::UnsignedIndex_t ndata = 0;
        for (int kk = k - a_nneigh; kk <= k + a_nneigh; ++kk) {
          for (int jj = j - a_nneigh; jj <= j + a_nneigh; ++jj) {
            for (int ii = i - a_nneigh; ii <= i + a_nneigh; ++ii) {
              // Trap center cell
              if (ii == i && jj == j && kk == k) {
                neighborhood.setCenterOfStencil(ndata);
              }
              cells[ndata] = IRL::RectangularCuboid::fromBoundingPts(
                  IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                  IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
              neighborhood.setMember(ndata, &cells[ndata],
                                     &a_liquid_volume_fraction(ii, jj, kk));
              // Increment counter
              ++ndata;
            }
          }
        }
        auto found_planar_separator = (*a_liquid_gas_interface)(i, j, k);
        // Now perform actual LVIRA and obtain interface PlanarSeparator
        (*a_liquid_gas_interface)(i, j, k) =
            reconstructionWithLVIRA3D(neighborhood, found_planar_separator);
      }
    }
  }
}

void updatePolygon(const Data<double>& a_liquid_volume_fraction,
                   const Data<IRL::PlanarSeparator>& a_liquid_gas_interface,
                   Data<IRL::Polygon>* a_interface_polygon) {
  const BasicMesh& mesh = a_liquid_gas_interface.getMesh();
  // Loop over cells in domain. Skip if cell is not mixed phase.
  for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) >
                IRL::global_constants::VF_HIGH) {
          continue;
        }
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        (*a_interface_polygon)(i, j, k) =
            IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                cell, a_liquid_gas_interface(i, j, k),
                a_liquid_gas_interface(i, j, k)[0]);
      }
    }
  }
}

std::array<double, 6> fitParaboloidToPLICHeights(
    const Data<IRL::Polygon>& a_polygon, const Data<double>& a_volume_fraction,
    const IRL::Pt& a_reference_point, const IRL::ReferenceFrame& a_frame,
    const int a_i, const int a_j, const int a_k, const int a_nneigh,
    const double a_width) {
  const BasicMesh& mesh = a_polygon.getMesh();
  const double meshsize = 1.0;  // mesh.dx();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
  const int ic(a_i), jc(a_j), kc(a_k);
  const IRL::Pt pref = a_reference_point;
  const auto frame = a_frame;

  for (int k = kc - a_nneigh; k <= kc + a_nneigh; ++k) {
    for (int j = jc - a_nneigh; j <= jc + a_nneigh; ++j) {
      for (int i = ic - a_nneigh; i <= ic + a_nneigh; ++i) {
        const IRL::UnsignedIndex_t shape =
            a_polygon(i, j, k).getNumberOfVertices();
        if (shape == 0) {
          continue;
        }
        // Local polygon normal and centroid
        IRL::Pt ploc = a_polygon(i, j, k).calculateCentroid();
        IRL::Normal nloc = a_polygon(i, j, k).calculateNormal();
        // if (frame[2] * nloc <= 0.0) {
        //   continue;
        // }
        ploc -= pref;
        ploc /= meshsize;
        const IRL::Pt tmp_pt = ploc;
        const IRL::Normal tmp_n = nloc;
        for (IRL::UnsignedIndex_t d = 0; d < 3; ++d) {
          ploc[d] = frame[d] * tmp_pt;
          nloc[d] = frame[d] * tmp_n;
        }
        // Plane coefficients
        Eigen::VectorXd reconstruction_plane_coeffs(3);
        reconstruction_plane_coeffs << -(ploc * nloc) / meshsize, nloc[0],
            nloc[1];
        reconstruction_plane_coeffs /= -nloc[2];
        // Integrals
        Eigen::VectorXd integrals = Eigen::VectorXd::Zero(6);
        double b_dot_sum = 0.0;
        for (IRL::UnsignedIndex_t v = 0; v < shape; ++v) {
          IRL::UnsignedIndex_t vn = (v + 1) % shape;
          IRL::Pt vert1 = a_polygon(i, j, k)[v];
          IRL::Pt vert2 = a_polygon(i, j, k)[vn];
          vert1 -= pref;
          vert2 -= pref;
          vert1 /= meshsize;
          vert2 /= meshsize;
          IRL::Pt tmp_pt1 = vert1;
          IRL::Pt tmp_pt2 = vert2;
          for (IRL::UnsignedIndex_t d = 0; d < 3; ++d) {
            vert1[d] = frame[d] * tmp_pt1;
            vert2[d] = frame[d] * tmp_pt2;
          }

          const double xv = vert1[0];
          const double yv = vert1[1];
          const double xvn = vert2[0];
          const double yvn = vert2[1];

          Eigen::VectorXd integral_to_add(6);
          integral_to_add << (xv * yvn - xvn * yv) / 2.0,
              (xv + xvn) * (xv * yvn - xvn * yv) / 6.0,
              (yv + yvn) * (xv * yvn - xvn * yv) / 6.0,
              (xv + xvn) * (xv * xv + xvn * xvn) * (yvn - yv) / 12.0,
              (yvn - yv) *
                  (3.0 * xv * xv * yv + xv * xv * yvn + 2.0 * xv * xvn * yv +
                   2.0 * xv * xvn * yvn + xvn * xvn * yv +
                   3.0 * xvn * xvn * yvn) /
                  24.0,
              (xv - xvn) * (yv + yvn) * (yv * yv + yvn * yvn) / 12.0;
          integrals += integral_to_add;
        }
        b_dot_sum += integrals.head(3).dot(reconstruction_plane_coeffs);

        // Get weighting
        const double gaussianweight =  // 1.0;
            a_width <= 0.0
                ? 1.0
                : wgauss(std::sqrt(static_cast<IRL::Vec3<double>>(ploc) *
                                   static_cast<IRL::Vec3<double>>(ploc)),
                         a_width);
        const double vfrac = a_volume_fraction(i, j, k);
        double vfrac_weight = 1.0;
        const double limit_vfrac = 0.1;
        if (vfrac < limit_vfrac) {
          vfrac_weight = 0.5 - 0.5 * std::cos(M_PI * vfrac / limit_vfrac);
        } else if (vfrac > 1.0 - limit_vfrac) {
          vfrac_weight =
              0.5 - 0.5 * std::cos(M_PI * (1.0 - vfrac) / limit_vfrac);
        }
        double ww = 1.0;
        ww *= gaussianweight;
        ww *= vfrac_weight;

        if (ww > 0.0) {
          A += ww * integrals * integrals.transpose();
          b += ww * integrals * b_dot_sum;
        }
      }
    }
  }
  Eigen::VectorXd sol = A.colPivHouseholderQr().solve(b);
  return std::array<double, 6>{
      {sol(0), sol(1), sol(2), sol(3), sol(4), sol(5)}};
}

std::array<double, 6> fitParaboloidToCentroids(
    const Data<IRL::Polygon>& a_polygon, const Data<double>& a_volume_fraction,
    const IRL::Pt& a_reference_point, const IRL::ReferenceFrame& a_frame,
    const int a_i, const int a_j, const int a_k, const int a_nneigh,
    const double a_width) {
  const BasicMesh& mesh = a_polygon.getMesh();
  const double meshsize = 1.0;  // mesh.dx();
  // const int ncells = std::pow(2 * a_nneigh + 1, 3);
  const int ncells =
      (2 * a_nneigh + 1) * (2 * a_nneigh + 1) * (2 * a_nneigh + 1);
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ncells, 6);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(ncells);
  const int ic(a_i), jc(a_j), kc(a_k);
  const IRL::Pt pref = a_reference_point;
  const auto frame = a_frame;

  IRL::UnsignedIndex_t ndata = 0;
  for (int k = kc - a_nneigh; k <= kc + a_nneigh; ++k) {
    for (int j = jc - a_nneigh; j <= jc + a_nneigh; ++j) {
      for (int i = ic - a_nneigh; i <= ic + a_nneigh; ++i) {
        if (a_polygon(i, j, k).getNumberOfVertices() == 0) {
          continue;
        }
        IRL::Pt ploc = a_polygon(i, j, k).calculateCentroid();
        const IRL::Normal nloc = a_polygon(i, j, k).calculateNormal();
        const double surf = a_polygon(i, j, k).calculateAbsoluteVolume() /
                            (meshsize * meshsize);
        const double normalproj = std::max(frame[2] * nloc, 0.0);
        // if (normalproj <= 0.0) continue;
        ploc -= pref;
        ploc /= meshsize;
        const IRL::Pt tmp_pt = ploc;
        for (IRL::UnsignedIndex_t d = 0; d < 3; ++d) {
          ploc[d] = frame[d] * tmp_pt;
        }
        const double gaussianweight =
            // 1.0;
            a_width <= 0.0
                ? 1.0
                : wgauss(std::sqrt(static_cast<IRL::Vec3<double>>(ploc) *
                                   static_cast<IRL::Vec3<double>>(ploc)),
                         a_width);

        const double vfrac = a_volume_fraction(i, j, k);
        double vfrac_weight = 1.0;
        const double limit_vfrac = 0.1;
        if (vfrac < limit_vfrac) {
          vfrac_weight = 0.5 - 0.5 * std::cos(M_PI * vfrac / limit_vfrac);
        } else if (vfrac > 1.0 - limit_vfrac) {
          vfrac_weight =
              0.5 - 0.5 * std::cos(M_PI * (1.0 - vfrac) / limit_vfrac);
        }
        double ww = 1.0;
        ww *= normalproj;
        ww *= surf;
        ww *= gaussianweight;
        ww *= vfrac_weight;

        if (ww > 0.0) {
          // Store least squares matrix and RHS
          A(ndata, 0) = std::sqrt(ww);
          A(ndata, 1) = 0.0;
          A(ndata, 2) = 0.0;
          A(ndata, 3) = std::sqrt(ww) * ploc[0] * ploc[0];
          A(ndata, 4) = std::sqrt(ww) * ploc[0] * ploc[1];
          A(ndata, 5) = std::sqrt(ww) * ploc[1] * ploc[1];
          b(ndata) = std::sqrt(ww) * ploc[2];
          // Increment counter
          ++ndata;
        }
      }
    }
  }
  A.conservativeResize(ndata, Eigen::NoChange);
  b.conservativeResize(ndata, Eigen::NoChange);
  Eigen::VectorXd sol = A.colPivHouseholderQr().solve(b);
  return std::array<double, 6>{
      // {sol(0), sol(1), sol(2), sol(3), sol(4), sol(5)}};
      {sol(0), 0.0, 0.0, sol(3), sol(4), sol(5)}};
}

void Jibben::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                               const double a_dt, const Data<double>& a_U,
                               const Data<double>& a_V, const Data<double>& a_W,
                               Data<IRL::Paraboloid>* a_interface) {
  const BasicMesh& mesh = a_U.getMesh();

  Data<IRL::PlanarSeparator> interface(&mesh);
  updateReconstructionELVIRA(a_liquid_volume_fraction, &interface);
  updateReconstructionLVIRA(a_liquid_volume_fraction, 1, &interface);
  Data<IRL::Polygon> polygon(&mesh);
  updatePolygon(a_liquid_volume_fraction, interface, &polygon);
  polygon.updateBorder();

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[0] -= mesh.lx();
        }
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[0] += mesh.lx();
        }
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[1] -= mesh.ly();
        }
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[1] += mesh.ly();
        }
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[2] -= mesh.lz();
        }
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[2] += mesh.lz();
        }
      }
    }
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } else if (a_liquid_volume_fraction(i, j, k) >
                   IRL::global_constants::VF_HIGH) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
          // continue;
        } else {
          const IRL::Normal norm_poly = polygon(i, j, k).calculateNormal();
          const double poly_area = polygon(i, j, k).calculateVolume();
          const IRL::Pt pref = polygon(i, j, k).calculateCentroid();
          IRL::ReferenceFrame fit_frame;
          int largest_dir = 0;
          if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[1]))
            largest_dir = 1;
          if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[2]))
            largest_dir = 2;
          if (largest_dir == 0)
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 1.0, 0.0));
          else if (largest_dir == 1)
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 0.0, 1.0));
          else
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(1.0, 0.0, 0.0));
          fit_frame[0].normalize();
          fit_frame[1] = crossProduct(norm_poly, fit_frame[0]);
          fit_frame[2] = norm_poly;
          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                                      mesh.z(k + 1));
          const IRL::Pt cell_center = 0.5 * (lower_cell_pt + upper_cell_pt);
          IRL::Paraboloid paraboloid;

          double sum_vfrac = 0.0;
          for (int kk = -1; kk < 2; ++kk) {
            for (int jj = -1; jj < 2; ++jj) {
              for (int ii = -1; ii < 2; ++ii) {
                sum_vfrac += a_liquid_volume_fraction(i + ii, j + jj, k + kk);
              }
            }
          }

          auto sol_fit =
              fitParaboloidToPLICHeights(polygon, a_liquid_volume_fraction,
                                         pref, fit_frame, i, j, k, 1, 0.0);
          const double a = sol_fit[0], b = sol_fit[1], c = sol_fit[2],
                       d = sol_fit[3], e = sol_fit[4], f = sol_fit[5];
          const double theta = 0.5 * std::atan2(e, (IRL::safelyTiny(d - f)));
          const double cos_t = std::cos(theta);
          const double sin_t = std::sin(theta);
          const double A =
              -(d * cos_t * cos_t + f * sin_t * sin_t + e * cos_t * sin_t);
          const double B =
              -(f * cos_t * cos_t + d * sin_t * sin_t - e * cos_t * sin_t);
          // Translation to coordinate system R' where aligned paraboloid
          // valid Translation is R' = {x' = x + u, y' = y + v, z' = z + w}
          const double denominator = IRL::safelyTiny(4.0 * d * f - e * e);
          const double u = (2.0 * b * f - c * e) / denominator;
          const double v = -(b * e - 2.0 * d * c) / denominator;
          const double w =
              -(a + (-b * b * f + b * c * e - c * c * d) / denominator);

          IRL::UnitQuaternion rotation(theta, fit_frame[2]);
          IRL::Pt datum =
              pref - u * fit_frame[0] - v * fit_frame[1] - w * fit_frame[2];
          auto new_frame = rotation * fit_frame;
          const double max_curvature_dx = 1.0;
          double a_coeff = A;
          double b_coeff = B;
          if (std::sqrt(u * u + v * v + w * w) > 10.0 * mesh.dx() ||
              std::fabs(A) * mesh.dx() > max_curvature_dx ||
              std::fabs(B) * mesh.dx() > max_curvature_dx) {
            paraboloid = IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);
          } else {
            if (fabs(a_coeff) < 1.0e-3) {
              a_coeff = std::copysign(1.0e-3, a_coeff);
            }
            if (fabs(b_coeff) < 1.0e-3) {
              b_coeff = std::copysign(1.0e-3, b_coeff);
            }
            paraboloid = IRL::Paraboloid(datum, new_frame, a_coeff, b_coeff);
          }

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                                              upper_cell_pt);
          IRL::ProgressiveDistanceSolverParaboloid<IRL::RectangularCuboid>
              solver_distance(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
                              paraboloid);

          if (solver_distance.getDistance() == -DBL_MAX) {
            paraboloid = IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);
            IRL::ProgressiveDistanceSolverParaboloid<IRL::RectangularCuboid>
                new_solver_distance(cell, a_liquid_volume_fraction(i, j, k),
                                    1.0e-14, paraboloid);
            if (new_solver_distance.getDistance() == -DBL_MAX) {
              (*a_interface)(i, j, k) =
                  IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);
            } else {
              auto new_datum =
                  IRL::Pt(paraboloid.getDatum() +
                          new_solver_distance.getDistance() * fit_frame[2]);
              paraboloid.setDatum(new_datum);
              (*a_interface)(i, j, k) = paraboloid;
            }
          } else {
            auto new_datum =
                IRL::Pt(paraboloid.getDatum() +
                        solver_distance.getDistance() * fit_frame[2]);
            paraboloid.setDatum(new_datum);
            (*a_interface)(i, j, k) = paraboloid;
          }
          //std::cout << paraboloid.getAlignedParaboloid().a() << std::endl;
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void Centroid::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                 const double a_dt, const Data<double>& a_U,
                                 const Data<double>& a_V,
                                 const Data<double>& a_W,
                                 Data<IRL::Paraboloid>* a_interface) {
  const BasicMesh& mesh = a_U.getMesh();

  Data<IRL::PlanarSeparator> interface(&mesh);
  updateReconstructionELVIRA(a_liquid_volume_fraction, &interface);
  updateReconstructionLVIRA(a_liquid_volume_fraction, 1, &interface);
  Data<IRL::Polygon> polygon(&mesh);
  updatePolygon(a_liquid_volume_fraction, interface, &polygon);
  polygon.updateBorder();

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[0] -= mesh.lx();
        }
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[0] += mesh.lx();
        }
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[1] -= mesh.ly();
        }
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[1] += mesh.ly();
        }
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[2] -= mesh.lz();
        }
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[2] += mesh.lz();
        }
      }
    }
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } else if (a_liquid_volume_fraction(i, j, k) >
                   IRL::global_constants::VF_HIGH) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
          // continue;
        } else {
          const IRL::Normal norm_poly = polygon(i, j, k).calculateNormal();
          const double poly_area = polygon(i, j, k).calculateVolume();
          const IRL::Pt pref = polygon(i, j, k).calculateCentroid();
          IRL::ReferenceFrame fit_frame;
          int largest_dir = 0;
          if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[1]))
            largest_dir = 1;
          if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[2]))
            largest_dir = 2;
          if (largest_dir == 0)
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 1.0, 0.0));
          else if (largest_dir == 1)
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 0.0, 1.0));
          else
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(1.0, 0.0, 0.0));
          fit_frame[0].normalize();
          fit_frame[1] = crossProduct(norm_poly, fit_frame[0]);
          fit_frame[2] = norm_poly;
          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                                      mesh.z(k + 1));
          const IRL::Pt cell_center = 0.5 * (lower_cell_pt + upper_cell_pt);
          IRL::Paraboloid paraboloid;

          double sum_vfrac = 0.0;
          for (int kk = -1; kk < 2; ++kk) {
            for (int jj = -1; jj < 2; ++jj) {
              for (int ii = -1; ii < 2; ++ii) {
                sum_vfrac += a_liquid_volume_fraction(i + ii, j + jj, k + kk);
              }
            }
          }

          if (std::fabs(sum_vfrac - 27.0 * 0.5) < 10.0 &&
              a_liquid_volume_fraction(i, j, k) > 1.0e-6 &&
              a_liquid_volume_fraction(i, j, k) < 1.0 - 1.0e-6) {
            auto sol_fit =
                fitParaboloidToCentroids(polygon, a_liquid_volume_fraction,
                                         pref, fit_frame, i, j, k, 1, 0.0);
            const double a = sol_fit[0], b = sol_fit[1], c = sol_fit[2],
                         d = sol_fit[3], e = sol_fit[4], f = sol_fit[5];
            const double theta = 0.5 * std::atan2(e, (IRL::safelyTiny(d - f)));
            const double cos_t = std::cos(theta);
            const double sin_t = std::sin(theta);
            const double A =
                -(d * cos_t * cos_t + f * sin_t * sin_t + e * cos_t * sin_t);
            const double B =
                -(f * cos_t * cos_t + d * sin_t * sin_t - e * cos_t * sin_t);
            // Translation to coordinate system R' where aligned paraboloid
            // valid Translation is R' = {x' = x + u, y' = y + v, z' = z + w}
            const double denominator = IRL::safelyTiny(4.0 * d * f - e * e);
            const double u = (2.0 * b * f - c * e) / denominator;
            const double v = -(b * e - 2.0 * d * c) / denominator;
            const double w =
                -(a + (-b * b * f + b * c * e - c * c * d) / denominator);

            IRL::UnitQuaternion rotation(theta, fit_frame[2]);
            IRL::Pt datum =
                pref - u * fit_frame[0] - v * fit_frame[1] - w * fit_frame[2];
            auto new_frame = rotation * fit_frame;
            const double max_curvature_dx = 1.0;
            double a_coeff = A;
            double b_coeff = B;
            if (std::sqrt(u * u + v * v + w * w) > 10.0 * mesh.dx() ||
                std::fabs(A) * mesh.dx() > max_curvature_dx ||
                std::fabs(B) * mesh.dx() > max_curvature_dx) {
              paraboloid = IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);
            } else {
              paraboloid = IRL::Paraboloid(datum, new_frame, a_coeff, b_coeff);
            }
          } else {
            paraboloid = IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);
          }

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                                              upper_cell_pt);
          IRL::ProgressiveDistanceSolverParaboloid<IRL::RectangularCuboid>
              solver_distance(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
                              paraboloid);
          if (solver_distance.getDistance() == -DBL_MAX) {
            (*a_interface)(i, j, k) =
                IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);
          } else {
            auto new_datum =
                IRL::Pt(paraboloid.getDatum() +
                        solver_distance.getDistance() * fit_frame[2]);
            paraboloid.setDatum(new_datum);
            (*a_interface)(i, j, k) = paraboloid;
          }
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void PLIC::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                             const double a_dt, const Data<double>& a_U,
                             const Data<double>& a_V, const Data<double>& a_W,
                             Data<IRL::Paraboloid>* a_interface) {
  const BasicMesh& mesh = a_U.getMesh();

  Data<IRL::PlanarSeparator> interface(&mesh);
  updateReconstructionELVIRA(a_liquid_volume_fraction, &interface);
  updateReconstructionLVIRA(a_liquid_volume_fraction, 1, &interface);
  Data<IRL::Polygon> polygon(&mesh);
  updatePolygon(a_liquid_volume_fraction, interface, &polygon);
  polygon.updateBorder();

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[0] -= mesh.lx();
        }
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[0] += mesh.lx();
        }
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[1] -= mesh.ly();
        }
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[1] += mesh.ly();
        }
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[2] -= mesh.lz();
        }
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        for (auto& pt : polygon(i, j, k)) {
          pt[2] += mesh.lz();
        }
      }
    }
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } else if (a_liquid_volume_fraction(i, j, k) >
                   IRL::global_constants::VF_HIGH) {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
          // continue;
        } else {
          const IRL::Normal norm_poly = polygon(i, j, k).calculateNormal();
          const IRL::Pt pref = polygon(i, j, k).calculateCentroid();
          IRL::ReferenceFrame fit_frame;
          int largest_dir = 0;
          if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[1]))
            largest_dir = 1;
          if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[2]))
            largest_dir = 2;
          if (largest_dir == 0)
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 1.0, 0.0));
          else if (largest_dir == 1)
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 0.0, 1.0));
          else
            fit_frame[0] = crossProduct(norm_poly, IRL::Normal(1.0, 0.0, 0.0));
          fit_frame[0].normalize();
          fit_frame[1] = crossProduct(norm_poly, fit_frame[0]);
          fit_frame[2] = norm_poly;
          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                                      mesh.z(k + 1));
          const IRL::Pt cell_center = 0.5 * (lower_cell_pt + upper_cell_pt);
          IRL::Paraboloid paraboloid;

          paraboloid = IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                                              upper_cell_pt);
          IRL::ProgressiveDistanceSolverParaboloid<IRL::RectangularCuboid>
              solver_distance(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
                              paraboloid);

          if (solver_distance.getDistance() == -DBL_MAX) {
            (*a_interface)(i, j, k) =
                IRL::Paraboloid(pref, fit_frame, 1.0e-3, -1.0e-3);
          } else {
            auto new_datum =
                IRL::Pt(paraboloid.getDatum() +
                        solver_distance.getDistance() * fit_frame[2]);
            paraboloid.setDatum(new_datum);
            (*a_interface)(i, j, k) = paraboloid;
          }
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void ML::getReconstruction(const Data<double>& a_liquid_volume_fraction, const Data<IRL::Pt>& a_liquid_centroid, Data<IRL::Paraboloid>* a_interface) 
{
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  auto t = IRL::trainer(2);

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) 
  {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) 
    {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) 
      {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) 
        {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } 
        else if (a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) 
        {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
        } 
        else {
          IRL::Paraboloid paraboloid;

          Mesh mesh2(3, 3, 3, 1);
          auto nx = mesh2.getNx();
          auto ny = mesh2.getNy();
          auto nz = mesh2.getNz();
          IRL::Pt lower_domain(0, 0, 0);
          IRL::Pt upper_domain(3, 3, 3);
          mesh2.setCellBoundaries(lower_domain, upper_domain);

          DataMesh<double> neighborhood(mesh2);
          DataMesh<IRL::Pt> neighborhood_centroid(mesh2);
          for (int kk = k - 1; kk < k + 2; ++kk) 
          {
            for (int jj = j - 1; jj < j + 2; ++jj) 
            {
              for (int ii = i - 1; ii < i + 2; ++ii) 
              {
                neighborhood(ii - i, jj - j, kk - k) = a_liquid_volume_fraction(ii, jj, kk);
                //cout << neighborhood(ii - i+1, jj - j+1, kk - k+1) << " ";
                neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1) = a_liquid_centroid(ii, jj, kk);
                if (neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[0]!=0)
                {
                  neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[0] = neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[0] - mesh.x(i) - 0.5;
                }
                if (neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[1]!=0)
                {
                  neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[1] = neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[1] - mesh.y(j) - 0.5;
                }
                if (neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[2]!=0)
                {
                  neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[2] = neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[2] - mesh.z(k) - 0.5;
                }
                //cout << neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[0] << " " << mesh.x(i) << std::endl;
                //cout << neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[1] << " ";
                //cout << neighborhood_centroid(ii - i+1, jj - j+1, kk - k+1)[2] << " ";
              }
            }
          }
          //paraboloid = t.use_model("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model.pt", neighborhood, neighborhood_centroid);
          std::cout << paraboloid.getAlignedParaboloid().a() << std::endl;
          (*a_interface)(i, j, k) = paraboloid;
        }
      }
    }
  }

  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void ML_norm::getReconstruction(const Data<double>& a_liquid_volume_fraction, const Data<IRL::Pt>& a_liquid_centroid, Data<IRL::Paraboloid>* a_interface) 
{
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  auto t = IRL::trainer(6);

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) 
  {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) 
    {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) 
      {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) 
        {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } 
        else if (a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) 
        {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
        } 
        else {
          IRL::Paraboloid paraboloid;

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
                local_liquid_volume_fraction(ii-i+1, jj-j+1, kk-k+1) = a_liquid_volume_fraction(ii, jj, kk);
                local_liquid_centroid(ii-i+1, jj-j+1, kk-k+1) = a_liquid_centroid(ii, jj, kk);
              }
            }
          }
          //paraboloid = t.use_model2("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model_2.pt", local_liquid_volume_fraction, local_liquid_centroid);
          IRL::Pt datum = paraboloid.getDatum();
          datum[0] = (datum[0] + 0.5)*mesh.dx()+mesh.x(i);
          datum[1] = (datum[1] + 0.5)*mesh.dx()+mesh.y(i);
          datum[2] = (datum[2] + 0.5)*mesh.dx()+mesh.z(i);
          paraboloid.setDatum(datum);

          (*a_interface)(i, j, k) = paraboloid;
        }
      }
    }
  }

  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void ML_QUAD::getReconstruction(const Data<double>& a_liquid_volume_fraction, const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
                                 const double a_dt, const Data<double>& a_U,
                                 const Data<double>& a_V,
                                 const Data<double>& a_W,
                                 Data<IRL::Paraboloid>* a_interface) {
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  // Loop over cells in domain. Skip if cell is not mixed phase.
  int count = 0;
  vector<double> fractions;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) 
        {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysBelow();
        } 
        else if (a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) 
        {
          (*a_interface)(i, j, k) = IRL::Paraboloid::createAlwaysAbove();
        } 
        else 
        {
          int tol = 0;
          if (a_liquid_volume_fraction(i-1, j, k) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i+1, j, k) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i, j-1, k) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i, j+1, k) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i, j, k-1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i, j, k+1) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i-1, j-1, k) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i-1, j+1, k) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i+1, j-1, k) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i+1, j+1, k) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i-1, j, k-1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i-1, j, k+1) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i+1, j, k-1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i+1, j, k+1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i, j-1, k-1) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i, j-1, k+1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i, j+1, k-1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i, j+1, k+1) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i-1, j-1, k-1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i+1, j-1, k-1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i-1, j+1, k-1) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i-1, j-1, k+1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i+1, j+1, k-1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i+1, j-1, k+1) < IRL::global_constants::VF_LOW){++tol;}
          if(a_liquid_volume_fraction(i-1, j+1, k+1) < IRL::global_constants::VF_LOW){++tol;} if(a_liquid_volume_fraction(i+1, j+1, k+1) < IRL::global_constants::VF_LOW){++tol;}
          if (tol >= 20)
          {
            ++count;
          }
          // Build surrounding stencil information.
          fractions.clear();

          bool flip = false;
          if (a_liquid_volume_fraction(i,j,k) > 0.5)
          {
            flip = true;
          }

          if (!flip)
          {
            for (int ii = i - 1; ii < i + 2; ++ii) {
              for (int jj = j - 1; jj < j + 2; ++jj) {
                for (int kk = k - 1; kk < k + 2; ++kk) {
                  if (ii > mesh.imax() && jj > mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj < mesh.jmin() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && jj > mesh.jmax() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && jj < mesh.jmin() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj > mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && jj < mesh.jmin() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj > mesh.jmax() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj < mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz()); 
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz()); 
                  }


                  else if (ii > mesh.imax() && jj > mesh.jmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), kk));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), jj, mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (jj > mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, mesh.jmin(), mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  

                  else if (ii > mesh.imax() && jj < mesh.jmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), kk));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), jj, mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (jj > mesh.jmax() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, mesh.jmin(), mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }


                  else if (ii < mesh.imin() && jj < mesh.jmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), kk));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), jj, mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (jj < mesh.jmin() && kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, mesh.jmax(), mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }


                  else if (ii < mesh.imin() && jj > mesh.jmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), kk));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), jj, mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (jj < mesh.jmin() && kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, mesh.jmax(), mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  
                  
                  else if (ii > mesh.imax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imin(), jj, kk));
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (jj > mesh.jmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, mesh.jmin(), kk));
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (kk > mesh.kmax())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, jj, mesh.kmin()));
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii < mesh.imin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(mesh.imax(), jj, kk));
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (jj < mesh.jmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, mesh.jmax(), kk));
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (kk < mesh.kmin())
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, jj, mesh.kmax()));
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }


                  else
                  {
                    fractions.push_back(a_liquid_volume_fraction(ii, jj, kk));
                    fractions.push_back((a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                }
              }
            }
          }
          else
          {
            for (int ii = i - 1; ii < i + 2; ++ii) {
              for (int jj = j - 1; jj < j + 2; ++jj) {
                for (int kk = k - 1; kk < k + 2; ++kk) {
                  if (ii > mesh.imax() && jj > mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), mesh.kmin()));
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj < mesh.jmin() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), mesh.kmax()));
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && jj > mesh.jmax() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), mesh.kmax()));
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && jj < mesh.jmin() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), mesh.kmin()));
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj > mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), mesh.kmin()));
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && jj < mesh.jmin() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), mesh.kmax()));
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj > mesh.jmax() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), mesh.kmax()));
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && jj < mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), mesh.kmin()));
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz()); 
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz()); 
                  }


                  else if (ii > mesh.imax() && jj > mesh.jmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), mesh.jmin(), kk));
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), jj, mesh.kmin()));
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (jj > mesh.jmax() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, mesh.jmin(), mesh.kmin()));
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  

                  else if (ii > mesh.imax() && jj < mesh.jmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), mesh.jmax(), kk));
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii > mesh.imax() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), jj, mesh.kmax()));
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (jj > mesh.jmax() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, mesh.jmin(), mesh.kmax()));
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }


                  else if (ii < mesh.imin() && jj < mesh.jmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), mesh.jmax(), kk));
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), jj, mesh.kmax()));
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }
                  else if (jj < mesh.jmin() && kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, mesh.jmax(), mesh.kmax()));
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }


                  else if (ii < mesh.imin() && jj > mesh.jmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), mesh.jmin(), kk));
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (ii < mesh.imin() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), jj, mesh.kmin()));
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (jj < mesh.jmin() && kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, mesh.jmax(), mesh.kmin()));
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  
                  
                  else if (ii > mesh.imax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imin(), jj, kk));
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imin(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, kk)[0] - mesh.xm(mesh.imin()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imin(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (jj > mesh.jmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, mesh.jmin(), kk));
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), kk)[1] - mesh.ym(mesh.jmin()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmin(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (kk > mesh.kmax())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, jj, mesh.kmin()));
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmin())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmin())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmin())[2] - mesh.zm(mesh.kmin()))/mesh.dz());
                  }
                  else if (ii < mesh.imin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(mesh.imax(), jj, kk));
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(mesh.imax(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, kk)[0] - mesh.xm(mesh.imax()))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(mesh.imax(), jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (jj < mesh.jmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, mesh.jmax(), kk));
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), kk)[1] - mesh.ym(mesh.jmax()))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, mesh.jmax(), kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                  else if (kk < mesh.kmin())
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, jj, mesh.kmax()));
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmax())[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmax())[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, jj, mesh.kmax())[2] - mesh.zm(mesh.kmax()))/mesh.dz());
                  }


                  else
                  {
                    fractions.push_back(1 - a_liquid_volume_fraction(ii, jj, kk));
                    fractions.push_back((a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                    fractions.push_back((a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii))/mesh.dx());
                    fractions.push_back((a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj))/mesh.dy());
                    fractions.push_back((a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk))/mesh.dz());
                  }
                }
              }
            }
          }
          auto sm = IRL::spatial_moments();
          std::vector<double> center = sm.get_mass_centers_all(&fractions);
          int direction = 0;
            if (center[0] < 0 && center[1] >= 0 && center[2] >= 0)
            {
                direction = 1;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] < 0 && center[2] >= 0)
            {
                direction = 2;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] >= 0 && center[2] < 0)
            {
                direction = 3;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] < 0 && center[2] >= 0)
            {
                direction = 4;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] >= 0 && center[2] < 0)
            {
                direction = 5;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] >= 0 && center[1] < 0 && center[2] < 0)
            {
                direction = 6;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                    }
                }
                }
            }
            else if (center[0] < 0 && center[1] < 0 && center[2] < 0)
            {
                direction = 7;
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (i == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(2*9+j*3+k)+0];
                        fractions[7*(2*9+j*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(2*9+j*3+k)+1];
                        fractions[7*(2*9+j*3+k)+1] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(2*9+j*3+k)+2];
                        fractions[7*(2*9+j*3+k)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(2*9+j*3+k)+3];
                        fractions[7*(2*9+j*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
                        fractions[7*(2*9+j*3+k)+4] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
                        fractions[7*(2*9+j*3+k)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
                        fractions[7*(2*9+j*3+k)+6] = temp;
                    }
                    else if (i == 1)
                    {
                        fractions[7*(i*9+j*3+k)+1] = -fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (j == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+2*3+k)+0];
                        fractions[7*(i*9+2*3+k)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+2*3+k)+1];
                        fractions[7*(i*9+2*3+k)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+2*3+k)+2];
                        fractions[7*(i*9+2*3+k)+2] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = fractions[7*(i*9+2*3+k)+3];
                        fractions[7*(i*9+2*3+k)+3] = temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
                        fractions[7*(i*9+2*3+k)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
                        fractions[7*(i*9+2*3+k)+5] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
                        fractions[7*(i*9+2*3+k)+6] = temp;
                    }
                    else if (j == 1)
                    {
                        fractions[7*(i*9+j*3+k)+2] = -fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
                    }
                    }
                }
                }
                for (int i = 0; i < 3; ++i)
                {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                    if (k == 0)
                    {
                        double temp = fractions[7*(i*9+j*3+k)+0];
                        fractions[7*(i*9+j*3+k)+0] = fractions[7*(i*9+j*3+2)+0];
                        fractions[7*(i*9+j*3+2)+0] = temp;
                        temp = fractions[7*(i*9+j*3+k)+1];
                        fractions[7*(i*9+j*3+k)+1] = fractions[7*(i*9+j*3+2)+1];
                        fractions[7*(i*9+j*3+2)+1] = temp;
                        temp = fractions[7*(i*9+j*3+k)+2];
                        fractions[7*(i*9+j*3+k)+2] = fractions[7*(i*9+j*3+2)+2];
                        fractions[7*(i*9+j*3+2)+2] = temp;
                        temp = fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+2)+3];
                        fractions[7*(i*9+j*3+2)+3] = -temp;
                        temp = fractions[7*(i*9+j*3+k)+4];
                        fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
                        fractions[7*(i*9+j*3+2)+4] = temp;
                        temp = fractions[7*(i*9+j*3+k)+5];
                        fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
                        fractions[7*(i*9+j*3+2)+5] = temp;
                        temp = fractions[7*(i*9+j*3+k)+6];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
                        fractions[7*(i*9+j*3+2)+6] = -temp;
                    }
                    else if (k == 1)
                    {
                        fractions[7*(i*9+j*3+k)+3] = -fractions[7*(i*9+j*3+k)+3];
                        fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
                    }
                  }
                }
              }
          }
          //std::cout << fractions << std::endl;
          auto normal = IRL::Normal();
          normal = t.get_normal(&fractions);

          auto axis = IRL::Normal();
          axis = t_axis.get_para_axis(&fractions);
          axis.normalize();
          fractions.insert(fractions.begin(),axis[2]);
          fractions.insert(fractions.begin(),axis[1]);
          fractions.insert(fractions.begin(),axis[0]);
          vector<double> coeff = t_coeff.get_para_coeff(&fractions);
          fractions.insert(fractions.begin(),coeff[1]);
          fractions.insert(fractions.begin(),coeff[0]);
          auto origin = IRL::Normal();
          origin = t_origin.get_para_origin(&fractions);

          switch (direction)
          {
              case 1:
              normal[0] = -normal[0];
              axis[0] = -axis[0];
              origin[0] = -origin[0];
              break;
              case 2:
              normal[1] = -normal[1];
              axis[1] = -axis[1];
              origin[1] = -origin[1];
              break;
              case 3:
              normal[2] = -normal[2];
              axis[2] = -axis[2];
              origin[2] = -origin[2];
              break;
              case 4:
              normal[0] = -normal[0];
              normal[1] = -normal[1];
              axis[0] = -axis[0];
              axis[1] = -axis[1];
              origin[0] = -origin[0];
              origin[1] = -origin[1];
              break;
              case 5:
              normal[0] = -normal[0];
              normal[2] = -normal[2];
              axis[0] = -axis[0];
              axis[2] = -axis[2];
              origin[0] = -origin[0];
              origin[2] = -origin[2];
              break;
              case 6:
              normal[1] = -normal[1];
              normal[2] = -normal[2];
              axis[1] = -axis[1];
              axis[2] = -axis[2];
              origin[1] = -origin[1];
              origin[2] = -origin[2];
              break;
              case 7:
              normal[0] = -normal[0];
              normal[1] = -normal[1];
              normal[2] = -normal[2];
              axis[0] = -axis[0];
              axis[1] = -axis[1];
              axis[2] = -axis[2];
              origin[0] = -origin[0];
              origin[1] = -origin[1];
              origin[2] = -origin[2];
              break;
          }
          if (!flip)
          {
              normal[0] = -normal[0];
              normal[1] = -normal[1];
              normal[2] = -normal[2];
              axis[0] = -axis[0];
              axis[1] = -axis[1];
              axis[2] = -axis[2];
              coeff[0] = -coeff[0];
              coeff[1] = -coeff[1];
          }
          coeff[0] = coeff[0]/mesh.dx();
          coeff[1] = coeff[1]/mesh.dx();
          std::cout << normal << std::endl;
          std::cout << axis << std::endl << std::endl;

          origin[0] = origin[0]*mesh.dx()+mesh.xm(i);
          origin[1] = origin[0]*mesh.dy()+mesh.ym(j);
          origin[2] = origin[0]*mesh.dy()+mesh.zm(k);

          double n2 = axis[0]/(sqrt(axis[1]*axis[1]+axis[0]*axis[0]));
          double n1 = (-n2*axis[1])/axis[0];

          IRL::Normal n = IRL::Normal(n1,n2,0.0);
          IRL::Normal b = IRL::crossProduct(n,axis);
          b[0]=-b[0];
          b[1]=-b[1];
          b[2]=-b[2];

          IRL::ReferenceFrame f = IRL::ReferenceFrame(n, b, axis);
          IRL::Pt datum = IRL::Pt(origin[0],origin[1],origin[2]);

          IRL::Paraboloid paraboloid = IRL::Paraboloid(datum, f, coeff[0], coeff[1]);

          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                                      mesh.z(k + 1));

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                                              upper_cell_pt);
          IRL::ProgressiveDistanceSolverParaboloid<IRL::RectangularCuboid>
              solver_distance(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
                              paraboloid);

          if (solver_distance.getDistance() == -DBL_MAX) {
            paraboloid = IRL::Paraboloid(datum, f, 1.0e-3, -1.0e-3);
            IRL::ProgressiveDistanceSolverParaboloid<IRL::RectangularCuboid>
                new_solver_distance(cell, a_liquid_volume_fraction(i, j, k),
                                    1.0e-14, paraboloid);
            if (new_solver_distance.getDistance() == -DBL_MAX) {
              (*a_interface)(i, j, k) =
                  IRL::Paraboloid(datum, f, 1.0e-3, -1.0e-3);
            } else {
              auto new_datum =
                  IRL::Pt(paraboloid.getDatum() +
                          new_solver_distance.getDistance() * f[2]);
              paraboloid.setDatum(new_datum);
              (*a_interface)(i, j, k) = paraboloid;
            }
          } else {
            auto new_datum =
                IRL::Pt(paraboloid.getDatum() +
                        solver_distance.getDistance() * f[2]);
            paraboloid.setDatum(new_datum);
            (*a_interface)(i, j, k) = paraboloid;
          }

          //(*a_interface)(i, j, k) = IRL::Paraboloid(datum, f, coeff[0], coeff[1]);
          //std::cout << (*a_interface)(i, j, k) << std::endl;
          //CONSERVE MASS
        }
      }
    }
  }
  if (count > 0)
  {
    std::cout << "\nSpurious Planes: " << count << std::endl;
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void correctInterfacePlaneBorders(Data<IRL::Paraboloid>* a_interface) {
  const BasicMesh& mesh = (*a_interface).getMesh();
  // Fix distances in reconstruction for periodic boundary

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        IRL::Pt datum = (*a_interface)(i, j, k).getDatum();
        datum[0] -= mesh.lx();
        (*a_interface)(i, j, k).setDatum(datum);
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        IRL::Pt datum = (*a_interface)(i, j, k).getDatum();
        datum[0] += mesh.lx();
        (*a_interface)(i, j, k).setDatum(datum);
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        IRL::Pt datum = (*a_interface)(i, j, k).getDatum();
        datum[1] -= mesh.ly();
        (*a_interface)(i, j, k).setDatum(datum);
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        IRL::Pt datum = (*a_interface)(i, j, k).getDatum();
        datum[1] += mesh.ly();
        (*a_interface)(i, j, k).setDatum(datum);
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) {
        IRL::Pt datum = (*a_interface)(i, j, k).getDatum();
        datum[2] -= mesh.lz();
        (*a_interface)(i, j, k).setDatum(datum);
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        IRL::Pt datum = (*a_interface)(i, j, k).getDatum();
        datum[2] += mesh.lz();
        (*a_interface)(i, j, k).setDatum(datum);
      }
    }
  }
}
