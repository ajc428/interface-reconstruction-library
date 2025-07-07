// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2022 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/cylinder_advector/reconstruction_types.h"

#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/interface_reconstruction_methods/constrained_optimization_behavior.h"
#include "irl/interface_reconstruction_methods/elvira_neighborhood.h"
#include "irl/interface_reconstruction_methods/plvira_neighborhood.h"
#include "irl/interface_reconstruction_methods/progressive_distance_solver_paraboloid.h"
#include "irl/interface_reconstruction_methods/progressive_radius_solver_cylinder.h"
#include "irl/interface_reconstruction_methods/reconstruction_interface.h"
//#include "irl/moments/volume_moments_with_gradient.h"
//#include "irl/moments/volume_with_gradient.h"
#include "irl/optimization/constrained_levenberg_marquardt.h"
//#include "irl/paraboloid_reconstruction/gradient_paraboloid.h"
//#include "irl/paraboloid_reconstruction/hessian_paraboloid.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "examples/cylinder_advector/basic_mesh.h"
#include "examples/cylinder_advector/data.h"
#include "examples/cylinder_advector/vof_advection.h"
//#include "irl/machine_learning_reconstruction/trainer.h"

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
    Jibben::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_dt, a_U, a_V, a_W,
                              a_interface);
  } else if (a_reconstruction_method == "PLIC") {
    PLIC::getReconstruction(a_liquid_volume_fraction, a_dt, a_U, a_V, a_W,
                            a_interface);
  } else {
    std::cout << "Unknown reconstruction method of : "
              << a_reconstruction_method << '\n';
    std::cout << "Valid entries are: PLIC, Jibben. \n";
    std::exit(-1);
  }
}

void getReconstruction(const std::string& a_reconstruction_method,
                       const Data<double>& a_liquid_volume_fraction,
                       const Data<IRL::Pt>& a_liquid_centroid,
                       const Data<IRL::Pt>& a_gas_centroid,
                       const Data<IRL::LocalizedCylinderLink<double>>&
                           a_localized_cylinder_link,
                       const double a_dt, const Data<double>& a_U,
                       const Data<double>& a_V, const Data<double>& a_W,
                       Data<IRL::Cylinder>* a_interface) {
  if (a_reconstruction_method == "Cylinder_PCA") {
    Cylinder_PCA::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid, a_dt, a_U, a_V, a_W,
                              a_interface);
  } else if (a_reconstruction_method == "Cylinder_Spline") {
    Cylinder_Spline::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid, a_dt, a_U, a_V, a_W,
                              a_interface);
  } else if (a_reconstruction_method == "Cylinder_Curve_Local") {
    Cylinder_Curve_Local::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid, a_dt, a_U, a_V, a_W,
                              a_interface);
  } else if (a_reconstruction_method == "Cylinder_Curve_Global") {
    Cylinder_Curve_Global::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid, a_dt, a_U, a_V, a_W,
                              a_interface);
  } else {
    std::cout << "Unknown reconstruction method of : "
              << a_reconstruction_method << '\n';
    std::cout << "Valid entries are: Cylinder_PCA, Cylinder_Spline, Cylinder_Curve_Local, Cylinder_Curve_Global. \n";
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

void Jibben::getReconstruction(const Data<double>& a_liquid_volume_fraction,const Data<IRL::Pt>& a_liquid_centroid,
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
//paraboloid = gradientDescent(paraboloid,cell,a_liquid_centroid(i, j, k),a_liquid_volume_fraction(i, j, k));
(*a_interface)(i, j, k) = paraboloid;
}
} else {
auto new_datum =
IRL::Pt(paraboloid.getDatum() +
solver_distance.getDistance() * fit_frame[2]);
paraboloid.setDatum(new_datum);
//paraboloid = gradientDescent(paraboloid,cell,a_liquid_centroid(i, j, k),a_liquid_volume_fraction(i, j, k));
(*a_interface)(i, j, k) = paraboloid;
}
//std::cout << paraboloid.getAlignedParaboloid().a() << std::endl;
//std::cout << paraboloid.getReferenceFrame()[2] << std::endl;
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


void Cylinder_PCA::getReconstruction(const Data<double>& b_liquid_volume_fraction,const Data<IRL::Pt>& b_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
  const double a_dt, const Data<double>& a_U,
  const Data<double>& a_V, const Data<double>& a_W,
  Data<IRL::Cylinder>* a_interface) 
{
  const BasicMesh& mesh = a_U.getMesh();
  Data<double> a_liquid_volume_fraction = b_liquid_volume_fraction;
  Data<IRL::Pt> a_liquid_centroid = b_liquid_centroid;

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imax(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imax(),j,k)[0] - mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imax(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imax(),j,k)[2];
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imin(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imin(),j,k)[0] + mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imin(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imin(),j,k)[2];
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmax(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmax(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmax(),k)[1] - mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmax(),k)[2];
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmin(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmin(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmin(),k)[1] + mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmin(),k)[2];
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmax());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmax())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmax())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmax())[2] - mesh.lz();
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmin());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmin())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmin())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmin())[2] + mesh.lz();
      }
    }
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) 
  {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) 
    {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) 
      {
        IRL::Cylinder cylinder;
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysBelow();
        } 
        else if (a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysAbove();
        } 
        else 
        {
          Eigen::MatrixXd bary(27,3);
          int count = 0;
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  ++count;
                }
              }
            }
          }
          bary.resize(count,3);

          count = 0;
          double VF = 0;
          IRL::Pt datum = IRL::Pt(0,0,0);
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  bary(count,0) = a_liquid_centroid(ii,jj,kk)[0]/mesh.dx();
                  bary(count,1) = a_liquid_centroid(ii,jj,kk)[1]/mesh.dy();
                  bary(count,2) = a_liquid_centroid(ii,jj,kk)[2]/mesh.dz();
                  ++count;
                  datum = datum + a_liquid_centroid(ii,jj,kk) * a_liquid_volume_fraction(ii,jj,kk);
                  VF = VF + a_liquid_volume_fraction(ii,jj,kk);
                }
              }
            }
          }
          datum = datum / VF;
          Eigen::MatrixXd centered = bary.rowwise() - bary.colwise().mean(); 
          Eigen::MatrixXd cov = (centered.transpose()*centered) / (centered.rows()-1);
          Eigen::EigenSolver<Eigen::MatrixXd> es(cov);
          Eigen::Index maxL;
          es.eigenvalues().real().maxCoeff(&maxL);
          Eigen::VectorXd dir = es.eigenvectors().real().col(maxL);
          IRL::Normal direction;
          direction[0] = dir(0);
          direction[1] = dir(1);
          direction[2] = dir(2);

          direction.normalize();
          double n3 = 0;
          double n2 = 0;
          double n1 = 0;
          IRL::Normal v1;
          if (abs(direction[0]) >= abs(direction[1]) && abs(direction[0]) >= abs(direction[2]))
          {
            double n2 = direction[0]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n1 = (-n2*direction[1])/direction[0];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[1]) >= abs(direction[0]) && abs(direction[1]) >= abs(direction[2]))
          {
            double n1 = direction[1]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n2 = (-n1*direction[0])/direction[1];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[2]) >= abs(direction[0]) && abs(direction[2]) >= abs(direction[1]))
          {
            double n2 = direction[2]/(sqrt(direction[1]*direction[1]+direction[2]*direction[2]));
            double n3 = (-n2*direction[1])/direction[2];
            v1[0] = 0; v1[1] = n2; v1[2] = n3;
          }
          else
          {
            v1[0] = 0; v1[1] = 0; v1[2] = 0;
          }
          IRL::Normal b = IRL::crossProduct(direction,v1);
          b.normalize();
          IRL::Normal a = IRL::crossProduct(b,direction);
          a.normalize();
          IRL::ReferenceFrame frame = IRL::ReferenceFrame(direction, a, b);

          cylinder = IRL::Cylinder(datum, frame, 1, 0.00025);

          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                  mesh.z(k + 1));

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                          upper_cell_pt);
          IRL::ProgressiveRadiusSolverCylinder<IRL::RectangularCuboid>
          solver_radius(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
          cylinder);

          cylinder = solver_radius.getCylinder();

          (*a_interface)(i, j, k) = cylinder;
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void Cylinder_Curve_Global::getReconstruction(const Data<double>& b_liquid_volume_fraction,const Data<IRL::Pt>& b_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
  const double a_dt, const Data<double>& a_U,
  const Data<double>& a_V, const Data<double>& a_W,
  Data<IRL::Cylinder>* a_interface) 
{
  const BasicMesh& mesh = a_U.getMesh();
  Data<double> a_liquid_volume_fraction = b_liquid_volume_fraction;
  Data<IRL::Pt> a_liquid_centroid = b_liquid_centroid;

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imax(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imax(),j,k)[0] - mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imax(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imax(),j,k)[2];
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imin(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imin(),j,k)[0] + mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imin(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imin(),j,k)[2];
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmax(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmax(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmax(),k)[1] - mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmax(),k)[2];
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmin(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmin(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmin(),k)[1] + mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmin(),k)[2];
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmax());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmax())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmax())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmax())[2] - mesh.lz();
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmin());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmin())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmin())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmin())[2] + mesh.lz();
      }
    }
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) 
  {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) 
    {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) 
      {
        IRL::Cylinder cylinder;
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysBelow();
        } 
        else if (a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysAbove();
        } 
        else 
        {
          Eigen::MatrixXd bary(27,3);
          Eigen::VectorXd VFs(27);
          int count = 0;
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  ++count;
                }
              }
            }
          }
          bary.resize(count,3);
          VFs.resize(count);

          count = 0;
          double VF = 0;
          IRL::Pt datum = IRL::Pt(0,0,0);
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  bary(count,0) = a_liquid_centroid(ii,jj,kk)[0];
                  bary(count,1) = a_liquid_centroid(ii,jj,kk)[1];
                  bary(count,2) = a_liquid_centroid(ii,jj,kk)[2];
                  VFs(count) = a_liquid_volume_fraction(ii,jj,kk);
                  ++count;
                  datum = datum + a_liquid_centroid(ii,jj,kk) * a_liquid_volume_fraction(ii,jj,kk);
                  VF = VF + a_liquid_volume_fraction(ii,jj,kk);
                }
              }
            }
          }
          datum = datum / VF;

          PrincipalCurve pc = PrincipalCurve(bary, VFs);
          pc.fit();
          Eigen::MatrixXd curve = pc.getCurve();
          IRL::Normal direction = IRL::Normal(1.0,0.0,0.0);
          pc.fitSpline(&direction, &datum, IRL::Pt(mesh.xm(i),mesh.ym(j),mesh.zm(k)));//a_liquid_centroid(i,j,k));//

          direction.normalize();
          double n3 = 0;
          double n2 = 0;
          double n1 = 0;
          IRL::Normal v1;
          if (abs(direction[0]) >= abs(direction[1]) && abs(direction[0]) >= abs(direction[2]))
          {
            double n2 = direction[0]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n1 = (-n2*direction[1])/direction[0];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[1]) >= abs(direction[0]) && abs(direction[1]) >= abs(direction[2]))
          {
            double n1 = direction[1]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n2 = (-n1*direction[0])/direction[1];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[2]) >= abs(direction[0]) && abs(direction[2]) >= abs(direction[1]))
          {
            double n2 = direction[2]/(sqrt(direction[1]*direction[1]+direction[2]*direction[2]));
            double n3 = (-n2*direction[1])/direction[2];
            v1[0] = 0; v1[1] = n2; v1[2] = n3;
          }
          else
          {
            v1[0] = 0; v1[1] = 0; v1[2] = 0;
          }
          IRL::Normal b = IRL::crossProduct(direction,v1);
          b.normalize();
          IRL::Normal a = IRL::crossProduct(b,direction);
          a.normalize();
          IRL::ReferenceFrame frame = IRL::ReferenceFrame(direction, a, b);

          cylinder = IRL::Cylinder(datum, frame, 1, 0.00025);

          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                  mesh.z(k + 1));

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                          upper_cell_pt);
          IRL::ProgressiveRadiusSolverCylinder<IRL::RectangularCuboid>
          solver_radius(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
          cylinder);

          cylinder = solver_radius.getCylinder();
          (*a_interface)(i, j, k) = cylinder;
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
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

void correctInterfacePlaneBorders(Data<IRL::Cylinder>* a_interface) {
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


PrincipalCurve::PrincipalCurve(const Eigen::MatrixXd& d, const Eigen::VectorXd& VFs)
{
  data = d;
  VF = VFs;
}

void PrincipalCurve::fit(int max_iterations, double tolerance) 
{
    initializeWithPCA();

    bool flag = true;
    int i = 0;
    while(flag && i < max_iterations)
    {
        Eigen::MatrixXd old_curve = curve;

        projectDataOntoCurve();
        updateCurve();
        //smoothCurve(3);
        orderCurvePoints();

        double change = (curve - old_curve).squaredNorm();
        //std::cout << "Iteration " << i + 1 << ", Change: " << change << std::endl;
        if (change < tolerance) 
        {
            //std::cout << "Converged!" << std::endl;
            flag = false;
        }
        ++i;
    }
}

const Eigen::MatrixXd& PrincipalCurve::getCurve() const 
{
    return curve;
}

void PrincipalCurve::initializeWithPCA() 
{
    Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean(); 
    Eigen::MatrixXd cov = (centered.transpose()*centered) / (centered.rows()-1);
    Eigen::EigenSolver<Eigen::MatrixXd> es(cov);
    Eigen::Index maxL;
    es.eigenvalues().real().maxCoeff(&maxL);
    Eigen::VectorXd dir = es.eigenvectors().real().col(maxL);

    Eigen::VectorXd projections = centered * dir;

    double min_proj = projections.minCoeff();
    double max_proj = projections.maxCoeff();
    
    curve = Eigen::MatrixXd(res, data.cols());
    for (int i = 0; i < res; ++i) 
    {
        double p = min_proj + (max_proj - min_proj) * i / (res-1);
        curve.row(i) = data.colwise().mean() + p * dir.transpose();
    }
    orderCurvePoints();
}

void PrincipalCurve::projectDataOntoCurve() 
{
    projection_indices.resize(data.rows());

    for (int i = 0; i < data.rows(); ++i) 
    {
        double min_dist_sq = -1.0;
        int best_idx = 0;

        for (int j = 0; j < curve.rows(); ++j) 
        {
            double dist_sq = (data.row(i) - curve.row(j)).squaredNorm();
            if (min_dist_sq < 0 || dist_sq < min_dist_sq) 
            {
                min_dist_sq = dist_sq;
                best_idx = j;
            }
        }
        projection_indices(i) = best_idx;
    }
}

void PrincipalCurve::updateCurve() 
{
    Eigen::MatrixXd new_curve = Eigen::MatrixXd::Zero(curve.rows(), curve.cols());
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(curve.rows());
    Eigen::VectorXd VF_total = Eigen::VectorXd::Zero(curve.rows());
    for (int i = 0; i < data.rows(); ++i) 
    {
        int idx = projection_indices(i);
        new_curve.row(idx) += data.row(i)*VF(i);
        counts(idx)++;
        VF_total(idx) +=  VF(i);
    }

    for (int i = 0; i < curve.rows(); ++i) 
    {
        if (counts(i) > 0) 
        {
            //curve.row(i) = new_curve.row(i) / counts(i);
            curve.row(i) = new_curve.row(i) / VF_total(i);
        }
    }
}

void PrincipalCurve::smoothCurve(int window_size) 
{
    if (window_size >= 2) 
    {
      Eigen::MatrixXd smoothed_curve = curve;
      int half_window = window_size / 2;

      for (int i = 0; i < curve.rows(); ++i) 
      {
          Eigen::RowVectorXd sum = Eigen::RowVectorXd::Zero(curve.cols());
          int count = 0;
          for (int j = -half_window; j <= half_window; ++j) 
          {
              int idx = i + j;
              if (idx >= 0 && idx < curve.rows()) 
              {
                  sum += curve.row(idx);
                  count++;
              }
          }
          if (count > 0) 
          {
              smoothed_curve.row(i) = sum / count;
          }
      }
      curve = smoothed_curve;
    }
}

void PrincipalCurve::orderCurvePoints() 
{
    lambda.resize(curve.rows());
    lambda(0) = 0.0;
    for (int i = 1; i < curve.rows(); ++i) 
    {
        lambda(i) = lambda(i - 1) + (curve.row(i) - curve.row(i - 1)).norm();
    }

    std::vector<int> indices(curve.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return lambda(a) < lambda(b); });

    Eigen::MatrixXd sorted_curve(curve.rows(), curve.cols());
    Eigen::VectorXd sorted_lambda(lambda.size());
    for (int i = 0; i < curve.rows(); ++i) 
    {
        sorted_curve.row(i) = curve.row(indices[i]);
        sorted_lambda(i) = lambda(indices[i]);
    }
    curve = sorted_curve;
    lambda = sorted_lambda;
}

void PrincipalCurve::fitSpline(IRL::Normal *direction, IRL::Pt *pt, IRL::Pt target)
{
    Eigen::MatrixXd points = curve; 
    Eigen::VectorXd t(curve.rows());
    for (int i = 0; i < curve.rows(); ++i)
    {
      t(i) = lambda(i) / lambda(curve.rows()-1);
    }
    IRL::Pt origin;
    double par = 0;
    double mag = 100;
    for (int i = 0; i < 100; ++i) 
    {
      double par2 = i/99.0;
      origin[0] = points(0,0)*(((par2-t(1))*(par2-t(2)))/((t(0)-t(1))*(t(0)-t(2))))+points(1,0)*(((par2-t(0))*(par2-t(2)))/((t(1)-t(0))*(t(1)-t(2))))+points(2,0)*(((par2-t(0))*(par2-t(1)))/((t(2)-t(0))*(t(2)-t(1))));
      origin[1] = points(0,1)*(((par2-t(1))*(par2-t(2)))/((t(0)-t(1))*(t(0)-t(2))))+points(1,1)*(((par2-t(0))*(par2-t(2)))/((t(1)-t(0))*(t(1)-t(2))))+points(2,1)*(((par2-t(0))*(par2-t(1)))/((t(2)-t(0))*(t(2)-t(1))));
      origin[2] = points(0,2)*(((par2-t(1))*(par2-t(2)))/((t(0)-t(1))*(t(0)-t(2))))+points(1,2)*(((par2-t(0))*(par2-t(2)))/((t(1)-t(0))*(t(1)-t(2))))+points(2,2)*(((par2-t(0))*(par2-t(1)))/((t(2)-t(0))*(t(2)-t(1))));
      double mag2 = pow(origin[0]-target[0],2.0)+pow(origin[1]-target[1],2.0)+pow(origin[2]-target[2],2.0);
      //double mag2 = pow(origin[0]-curve(1,0),2.0)+pow(origin[1]-curve(1,1),2.0)+pow(origin[2]-curve(1,2),2.0);
      if (mag2 < mag)
      {
        mag = mag2;
        par = par2;
      }
    }
    //par=0.5;
    pt[0][0] = points(0,0)*(((par-t(1))*(par-t(2)))/((t(0)-t(1))*(t(0)-t(2))))+points(1,0)*(((par-t(0))*(par-t(2)))/((t(1)-t(0))*(t(1)-t(2))))+points(2,0)*(((par-t(0))*(par-t(1)))/((t(2)-t(0))*(t(2)-t(1))));
    pt[0][1] = points(0,1)*(((par-t(1))*(par-t(2)))/((t(0)-t(1))*(t(0)-t(2))))+points(1,1)*(((par-t(0))*(par-t(2)))/((t(1)-t(0))*(t(1)-t(2))))+points(2,1)*(((par-t(0))*(par-t(1)))/((t(2)-t(0))*(t(2)-t(1))));
    pt[0][2] = points(0,2)*(((par-t(1))*(par-t(2)))/((t(0)-t(1))*(t(0)-t(2))))+points(1,2)*(((par-t(0))*(par-t(2)))/((t(1)-t(0))*(t(1)-t(2))))+points(2,2)*(((par-t(0))*(par-t(1)))/((t(2)-t(0))*(t(2)-t(1))));

    points = curve.rowwise() - curve.colwise().mean(); 
    direction[0][0] = points(0,0)*((2*par-t(1)-t(2))/((t(0)-t(1))*(t(0)-t(2))))+points(1,0)*((2*par-t(0)-t(2))/((t(1)-t(0))*(t(1)-t(2))))+points(2,0)*((2*par-t(0)-t(1))/((t(2)-t(0))*(t(2)-t(1))));
    direction[0][1] = points(0,1)*((2*par-t(1)-t(2))/((t(0)-t(1))*(t(0)-t(2))))+points(1,1)*((2*par-t(0)-t(2))/((t(1)-t(0))*(t(1)-t(2))))+points(2,1)*((2*par-t(0)-t(1))/((t(2)-t(0))*(t(2)-t(1))));
    direction[0][2] = points(0,2)*((2*par-t(1)-t(2))/((t(0)-t(1))*(t(0)-t(2))))+points(1,2)*((2*par-t(0)-t(2))/((t(1)-t(0))*(t(1)-t(2))))+points(2,2)*((2*par-t(0)-t(1))/((t(2)-t(0))*(t(2)-t(1))));
}










///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////











void Cylinder_Spline::getReconstruction(const Data<double>& b_liquid_volume_fraction,const Data<IRL::Pt>& b_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
  const double a_dt, const Data<double>& a_U,
  const Data<double>& a_V, const Data<double>& a_W,
  Data<IRL::Cylinder>* a_interface) 
{
  const BasicMesh& mesh = a_U.getMesh();
  Data<double> a_liquid_volume_fraction = b_liquid_volume_fraction;
  Data<IRL::Pt> a_liquid_centroid = b_liquid_centroid;

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imax(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imax(),j,k)[0] - mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imax(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imax(),j,k)[2];
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imin(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imin(),j,k)[0] + mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imin(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imin(),j,k)[2];
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmax(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmax(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmax(),k)[1] - mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmax(),k)[2];
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmin(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmin(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmin(),k)[1] + mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmin(),k)[2];
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmax());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmax())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmax())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmax())[2] - mesh.lz();
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmin());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmin())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmin())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmin())[2] + mesh.lz();
      }
    }
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) 
  {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) 
    {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) 
      {
        IRL::Cylinder cylinder;
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysBelow();
        } 
        else if (a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysAbove();
        } 
        else 
        {
          Eigen::MatrixXd bary(27,3);
          int count = 0;
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  ++count;
                }
              }
            }
          }
          bary.resize(count,3);

          count = 0;
          double VF = 0;
          IRL::Pt datum = IRL::Pt(0,0,0);
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  bary(count,0) = a_liquid_centroid(ii,jj,kk)[0]/mesh.dx();
                  bary(count,1) = a_liquid_centroid(ii,jj,kk)[1]/mesh.dy();
                  bary(count,2) = a_liquid_centroid(ii,jj,kk)[2]/mesh.dz();
                  ++count;
                  datum = datum + a_liquid_centroid(ii,jj,kk) * a_liquid_volume_fraction(ii,jj,kk);
                  VF = VF + a_liquid_volume_fraction(ii,jj,kk);
                }
              }
            }
          }
          datum = datum / VF;
          Eigen::MatrixXd centered = bary.rowwise() - bary.colwise().mean(); 
          IRL::Normal direction;
          
          Eigen::VectorXd dist(count);
          dist(0) = 0;
          for (int ii = 1; ii < count; ++ii)
          {
            dist(ii) = (centered.row(ii) - centered.row(ii-1)).norm() + dist(ii-1);
          }
          Eigen::VectorXd t(count);
          for (int ii = 0; ii < count; ++ii)
          {
            t(ii) = dist(ii) / dist(count-1);
          }

          // Eigen::MatrixXd T(count,4);
          // T.col(0) = t.array().pow(3.0);
          // T.col(1) = t.array().pow(2.0);
          // T.col(2) = t;
          // T.col(3) = Eigen::VectorXd::Ones(count);

          // Eigen::MatrixXd M(4,4);
          // M(0,0) = -1;
          // M(1,0) = 3;
          // M(2,0) = -3;
          // M(3,0) = 1;
          // M(0,1) = 3;
          // M(1,1) = -6;
          // M(2,1) = 3;
          // M(3,1) = 0;
          // M(0,2) = -3;
          // M(1,2) = 3;
          // M(2,2) = 0;
          // M(3,2) = 0;
          // M(0,3) = 1;
          // M(1,3) = 0;
          // M(2,3) = 0;
          // M(3,3) = 0;

          // Eigen::MatrixXd T_prime(count,4);
          // T_prime.col(0) = Eigen::VectorXd::Zero(count);
          // T_prime.col(1) = Eigen::VectorXd::Zero(count);
          // T_prime.col(2) = t;
          // T_prime.col(3) = Eigen::VectorXd::Ones(count);

          // Eigen::MatrixXd M_prime(4,4);
          // M_prime(0,0) = 0;
          // M_prime(1,0) = 0;
          // M_prime(2,0) = 0;
          // M_prime(3,0) = 0;
          // M_prime(0,1) = 0;
          // M_prime(1,1) = 0;
          // M_prime(2,1) = 0;
          // M_prime(3,1) = 0;
          // M_prime(0,2) = -6;
          // M_prime(1,2) = 18;
          // M_prime(2,2) = -18;
          // M_prime(3,2) = 6;
          // M_prime(0,3) = 6;
          // M_prime(1,3) = -12;
          // M_prime(2,3) = 6;
          // M_prime(3,3) = 0;

          // double lambda = 0;
          // Eigen::VectorXd P_x = (M.transpose()*T.transpose()*T*M+lambda*M_prime.transpose()*T_prime.transpose()*T_prime*M_prime).inverse()*M.transpose()*T.transpose()*centered.col(0);
          // Eigen::VectorXd P_y = (M.transpose()*T.transpose()*T*M+lambda*M_prime.transpose()*T_prime.transpose()*T_prime*M_prime).inverse()*M.transpose()*T.transpose()*centered.col(1);
          // Eigen::VectorXd P_z = (M.transpose()*T.transpose()*T*M+lambda*M_prime.transpose()*T_prime.transpose()*T_prime*M_prime).inverse()*M.transpose()*T.transpose()*centered.col(2);

          // // Eigen::VectorXd P_x = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(0);
          // // Eigen::VectorXd P_y = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(1);
          // // Eigen::VectorXd P_z = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(2);
          // // Eigen::VectorXd P_x = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(0);
          // // Eigen::VectorXd P_y = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(1);
          // // Eigen::VectorXd P_z = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(2);
          // Eigen::MatrixXd p(4,3);
          // p.col(0) = P_x; 
          // p.col(1) = P_y; 
          // p.col(2) = P_z; 

          // double par = 0;
          // double d = 100;
          // for (int ii = 0; ii <= 100; ++ii)
          // {
          //   Eigen::VectorXd tT(4);
          //   tT(0) = (ii/100.0)*(ii/100.0)*(ii/100.0);
          //   tT(1) = (ii/100.0)*(ii/100.0);
          //   tT(2) = (ii/100.0);
          //   tT(3) = 1;
          //   Eigen::VectorXd loc = tT.transpose()*M*p;
          //   double temp = loc.norm();
          //   if (temp < d)
          //   {
          //     d = temp;
          //     par = ii/100.0;
          //   }
          // }

          // direction[0] = -3*P_x(0)*(1-par)*(1-par) + 3*P_x(1)*(1-4*par+3*par*par) + 3*P_x(2)*par*(2-3*par) + 3*P_x(3)*par*par;
          // direction[1] = -3*P_y(0)*(1-par)*(1-par) + 3*P_y(1)*(1-4*par+3*par*par) + 3*P_y(2)*par*(2-3*par) + 3*P_y(3)*par*par;
          // direction[2] = -3*P_z(0)*(1-par)*(1-par) + 3*P_z(1)*(1-4*par+3*par*par) + 3*P_z(2)*par*(2-3*par) + 3*P_z(3)*par*par;



          Eigen::MatrixXd T(count,3);
          T.col(0) = t.array().pow(2.0);
          T.col(1) = t;
          T.col(2) = Eigen::VectorXd::Ones(count);

          Eigen::MatrixXd M(3,3);
          M(0,0) = 1;
          M(1,0) = -2;
          M(2,0) = -1;
          M(0,1) = -2;
          M(1,1) = 2;
          M(2,1) = 0;
          M(0,2) = 1;
          M(1,2) = 0;
          M(2,2) = 0;

          Eigen::MatrixXd T_prime(count,3);
          T_prime.col(0) = Eigen::VectorXd::Zero(count);
          T_prime.col(1) = Eigen::VectorXd::Zero(count);
          T_prime.col(2) = Eigen::VectorXd::Ones(count);

          Eigen::MatrixXd M_prime(3,3);
          M_prime(0,0) = 0;
          M_prime(1,0) = 0;
          M_prime(2,0) = 0;
          M_prime(0,1) = 0;
          M_prime(1,1) = 0;
          M_prime(2,1) = 0;
          M_prime(0,2) = 2;
          M_prime(1,2) = -4;
          M_prime(2,2) = -2;

          double lambda = 0;
          Eigen::VectorXd P_x = (M.transpose()*T.transpose()*T*M+lambda*M_prime.transpose()*T_prime.transpose()*T_prime*M_prime).inverse()*M.transpose()*T.transpose()*centered.col(0);
          Eigen::VectorXd P_y = (M.transpose()*T.transpose()*T*M+lambda*M_prime.transpose()*T_prime.transpose()*T_prime*M_prime).inverse()*M.transpose()*T.transpose()*centered.col(1);
          Eigen::VectorXd P_z = (M.transpose()*T.transpose()*T*M+lambda*M_prime.transpose()*T_prime.transpose()*T_prime*M_prime).inverse()*M.transpose()*T.transpose()*centered.col(2);

          // Eigen::VectorXd P_x = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(0);
          // Eigen::VectorXd P_y = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(1);
          // Eigen::VectorXd P_z = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(2);
          // Eigen::VectorXd P_x = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(0);
          // Eigen::VectorXd P_y = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(1);
          // Eigen::VectorXd P_z = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(2);
          Eigen::MatrixXd p(3,3);
          p.col(0) = P_x; 
          p.col(1) = P_y; 
          p.col(2) = P_z; 

          std::cout << p << std::endl << std::endl;

          double par = 0;
          double d = 100;
          Eigen::VectorXd point;
          for (int ii = 0; ii <= 100; ++ii)
          {
            Eigen::VectorXd tT(3);
            tT(0) = (ii/100.0)*(ii/100.0);
            tT(1) = (ii/100.0);
            tT(2) = 1;
            Eigen::VectorXd loc = tT.transpose()*M*p;
            double temp = loc.norm();
            if (temp < d)
            {
              d = temp;
              par = ii/100.0;
              point = loc;
            }
          }
          //std::cout << point << std::endl << std::endl;
          direction[0] = par*(2*P_x(0)-4*P_x(1)-2*P_x(2)) + (-2*P_x(0)+2*P_x(1));
          direction[1] = par*(2*P_y(0)-4*P_y(1)-2*P_y(2)) + (-2*P_y(0)+2*P_y(1));
          direction[2] = par*(2*P_z(0)-4*P_z(1)-2*P_z(2)) + (-2*P_z(0)+2*P_z(1));
          std::cout << direction << std::endl << std::endl;






          // Eigen::MatrixXd T(count,2);
          // T.col(0) = t;
          // T.col(1) = Eigen::VectorXd::Ones(count);

          // Eigen::MatrixXd M(2,2);
          // M(0,0) = -1;
          // M(1,0) = 1;
          // M(0,1) = 1;
          // M(1,1) = 0;

          // Eigen::VectorXd P_x = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(0);
          // Eigen::VectorXd P_y = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(1);
          // Eigen::VectorXd P_z = M.inverse() * (T.transpose()*T).inverse() * T.transpose() * centered.col(2);
          // // Eigen::VectorXd P_x = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(0);
          // // Eigen::VectorXd P_y = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(1);
          // // Eigen::VectorXd P_z = (M.transpose()*T.transpose()*T*M).inverse() * M.transpose() * T.transpose() * centered.col(2);
          // Eigen::MatrixXd p(2,3);
          // p.col(0) = P_x; 
          // p.col(1) = P_y; 
          // p.col(2) = P_z; 

          // std::cout << p << std::endl << std::endl;

          // double par = 0;
          // double d = 100;
          // Eigen::VectorXd point;
          // for (int ii = 0; ii <= 100; ++ii)
          // {
          //   Eigen::VectorXd tT(2);
          //   tT(0) = (ii/100.0);
          //   tT(1) = 1;
          //   Eigen::VectorXd loc = tT.transpose()*M*p;
          //   double temp = loc.norm();
          //   if (temp < d)
          //   {
          //     d = temp;
          //     par = ii/100.0;
          //     point = loc;
          //   }
          // }
          // //std::cout << point << std::endl << std::endl;
          // direction[0] = P_x(1) - P_x(0);
          // direction[1] = P_y(1) - P_y(0);
          // direction[2] = P_z(1) - P_z(0);
          // std::cout << direction << std::endl << std::endl;

          direction.normalize();
          std::cout << direction << std::endl << std::endl << std::endl << std::endl;
          double n3 = 0;
          double n2 = 0;
          double n1 = 0;
          IRL::Normal v1;
          if (abs(direction[0]) >= abs(direction[1]) && abs(direction[0]) >= abs(direction[2]))
          {
            double n2 = direction[0]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n1 = (-n2*direction[1])/direction[0];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[1]) >= abs(direction[0]) && abs(direction[1]) >= abs(direction[2]))
          {
            double n1 = direction[1]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n2 = (-n1*direction[0])/direction[1];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[2]) >= abs(direction[0]) && abs(direction[2]) >= abs(direction[1]))
          {
            double n2 = direction[2]/(sqrt(direction[1]*direction[1]+direction[2]*direction[2]));
            double n3 = (-n2*direction[1])/direction[2];
            v1[0] = 0; v1[1] = n2; v1[2] = n3;
          }
          else
          {
            v1[0] = 0; v1[1] = 0; v1[2] = 0;
          }
          IRL::Normal b = IRL::crossProduct(direction,v1);
          b.normalize();
          IRL::Normal a = IRL::crossProduct(b,direction);
          a.normalize();
          IRL::ReferenceFrame frame = IRL::ReferenceFrame(direction, a, b);

          cylinder = IRL::Cylinder(datum, frame, 1, 0.00025);

          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                  mesh.z(k + 1));

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                          upper_cell_pt);
          IRL::ProgressiveRadiusSolverCylinder<IRL::RectangularCuboid>
          solver_radius(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
          cylinder);

          cylinder = solver_radius.getCylinder();

          (*a_interface)(i, j, k) = cylinder;
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

bool myfunction (Eigen::VectorXd i,Eigen::VectorXd j) { return (i.sum()<j.sum()); }

void Cylinder_Curve_Local::getReconstruction(const Data<double>& b_liquid_volume_fraction,const Data<IRL::Pt>& b_liquid_centroid,const Data<IRL::Pt>& a_gas_centroid,
  const double a_dt, const Data<double>& a_U,
  const Data<double>& a_V, const Data<double>& a_W,
  Data<IRL::Cylinder>* a_interface) 
{
  const BasicMesh& mesh = a_U.getMesh();
  Data<double> a_liquid_volume_fraction = b_liquid_volume_fraction;
  Data<IRL::Pt> a_liquid_centroid = b_liquid_centroid;

  // x- boundary
  for (int i = mesh.imino(); i < mesh.imin(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imax(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imax(),j,k)[0] - mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imax(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imax(),j,k)[2];
      }
    }
  }

  // x+ boundary
  for (int i = mesh.imax() + 1; i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(mesh.imin(), j, k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(mesh.imin(),j,k)[0] + mesh.lx();
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(mesh.imin(),j,k)[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(mesh.imin(),j,k)[2];
      }
    }
  }

  // y- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j < mesh.jmin(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmax(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmax(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmax(),k)[1] - mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmax(),k)[2];
      }
    }
  }

  // y+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmax() + 1; j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, mesh.jmin(), k);
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,mesh.jmin(),k)[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,mesh.jmin(),k)[1] + mesh.ly();
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,mesh.jmin(),k)[2];
      }
    }
  }

  // z- boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) 
  {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) 
    {
      for (int k = mesh.kmino(); k < mesh.kmin(); ++k) 
      {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmax());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmax())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmax())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmax())[2] - mesh.lz();
      }
    }
  }

  // z+ boundary
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmax() + 1; k <= mesh.kmaxo(); ++k) {
        a_liquid_volume_fraction(i, j, k) = a_liquid_volume_fraction(i, j, mesh.kmin());
        a_liquid_centroid(i,j,k)[0] = a_liquid_centroid(i,j,mesh.kmin())[0];
        a_liquid_centroid(i,j,k)[1] = a_liquid_centroid(i,j,mesh.kmin())[1];
        a_liquid_centroid(i,j,k)[2] = a_liquid_centroid(i,j,mesh.kmin())[2] + mesh.lz();
      }
    }
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) 
  {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) 
    {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) 
      {
        IRL::Cylinder cylinder;
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysBelow();
        } 
        else if (a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) 
        {
          (*a_interface)(i, j, k) = IRL::Cylinder::createAlwaysAbove();
        } 
        else 
        {
          Eigen::MatrixXd bary(27,3);
          int count = 0;
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  ++count;
                }
              }
            }
          }
          bary.resize(count,3);

          count = 0;
          double VF = 0;
          IRL::Pt datum = IRL::Pt(0,0,0);
          for (int ii = i-1; ii <= i+1; ++ii) 
          {
            for (int jj = j-1; jj <= j+1; ++jj) 
            {
              for (int kk = k-1; kk <= k+1; ++kk) 
              {
                if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                {
                  //std::cout << a_liquid_centroid(ii,jj,kk) << std::endl;
                  bary(count,0) = a_liquid_centroid(ii,jj,kk)[0]/mesh.dx();
                  bary(count,1) = a_liquid_centroid(ii,jj,kk)[1]/mesh.dy();
                  bary(count,2) = a_liquid_centroid(ii,jj,kk)[2]/mesh.dz();
                  ++count;
                  datum = datum + a_liquid_centroid(ii,jj,kk) * a_liquid_volume_fraction(ii,jj,kk);
                  VF = VF + a_liquid_volume_fraction(ii,jj,kk);
                }
              }
            }
          }
          datum = datum / VF;
          Eigen::MatrixXd centered = bary.rowwise() - bary.colwise().mean(); 

          Eigen::VectorXd x = centered.row(0);
          // Eigen::VectorXd x(3);
          Eigen::VectorXd x_origin(3);
          // x(0) = datum[0]/mesh.dx() - bary.colwise().mean()(0);
          // x(1) = datum[1]/mesh.dy() - bary.colwise().mean()(1);
          // x(2) = datum[2]/mesh.dz() - bary.colwise().mean()(2);
          x_origin = x;

          Eigen::MatrixXd local(count,3);
          Eigen::VectorXd dir = Eigen::VectorXd::Zero(3);
          std::vector<Eigen::VectorXd> mus;
          Eigen::VectorXd mu = Eigen::VectorXd::Zero(3);
          bool flag = true;
          while (flag)
          {
            mu = Eigen::VectorXd::Zero(3);
            count = 0;
            int count1 = 0;
            VF = 0;
            for (int ii = i-1; ii <= i+1; ++ii) 
            {
              for (int jj = j-1; jj <= j+1; ++jj) 
              {
                for (int kk = k-1; kk <= k+1; ++kk) 
                {
                  if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                  {
                    //std::cout << count << " " << (x-centered.row(count)).norm() << std::endl;
                    if ((x-centered.row(count)).norm() <= 1.01)
                    {
                      double v = a_liquid_volume_fraction(ii,jj,kk);
                      mu(0) = mu(0) + centered.row(count)(0) * v;
                      mu(1) = mu(1) + centered.row(count)(1) * v;
                      mu(2) = mu(2) + centered.row(count)(2) * v;
                      VF = VF + v;
                      ++count1;
                    }
                    ++count;
                  }
                }
              }
            }
            //std::cout << count1 << std::endl;
            if (VF > IRL::global_constants::VF_LOW)
            {
              mu = mu / VF;
            }
            //std::cout << mu << std::endl;
            //std::cout << count1 << std::endl;
            // if (mus.size() > 0 && (mu - mus[mus.size()-1]).norm() < 0.001)
            // {
            //   flag = false;
            // }
            if (count1 <= 1)
            {
              flag = false;
            }
            else if (abs(x(0)) > 1.5 || abs(x(1)) > 1.5 || abs(x(2)) > 1.5)
            {
              flag = false;
            }
            else
            {
              if ((x - x_origin).norm() < 1e-5 || (mus.size() > 0 && (mu - mus[mus.size()-1]).norm() > 0.001))
              {
                mus.push_back(mu);
              }

              local.resize(count1, 3);
              count1 = 0;
              count = 0;
              for (int ii = i-1; ii <= i+1; ++ii) 
              {
                for (int jj = j-1; jj <= j+1; ++jj) 
                {
                  for (int kk = k-1; kk <= k+1; ++kk) 
                  {
                    if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                    {
                      if ((x-centered.row(count)).norm() <= 1.01)
                      {
                        local.row(count1) = centered.row(count);
                        ++count1;
                      }
                      ++count;
                    }
                  }
                }
              }
              Eigen::MatrixXd cov = (local.transpose()*local) / (local.rows()-1);
              Eigen::EigenSolver<Eigen::MatrixXd> es(cov);
              Eigen::Index maxL;
              es.eigenvalues().real().maxCoeff(&maxL);
              Eigen::VectorXd dir_new = es.eigenvectors().real().col(maxL);
              // std::cout << bary << std::endl << std::endl;
              //std::cout << centered << std::endl << std::endl;
              //std::cout << local << std::endl << std::endl;
              // std::cout << dir_new << std::endl << std::endl;
              dir_new.normalize();
              //std::cout << dir_new << std::endl << std::endl << std::endl << std::endl;
              if (dir_new.dot(dir) < 0)
              {
                dir_new = -dir_new;
              }
              dir = dir_new;
              //std::cout << mu << std::endl << std::endl;
              //std::cout << dir << std::endl << std::endl;
              x(0) = x(0) + 1.0*dir(0);
              x(1) = x(1) + 1.0*dir(1);
              x(2) = x(2) + 1.0*dir(2);
              //std::cout << x << std::endl << std::endl << std::endl << std::endl;
            }
          }
          // x(0) = datum[0]/mesh.dx() - bary.colwise().mean()(0);
          // x(1) = datum[1]/mesh.dy() - bary.colwise().mean()(1);
          // x(2) = datum[2]/mesh.dz() - bary.colwise().mean()(2);
          x = centered.row(0);
          local = Eigen::MatrixXd::Zero(count,3);
          dir = Eigen::VectorXd::Zero(3);
          flag = true;
          while (flag)
          {
            Eigen::VectorXd mu = Eigen::VectorXd::Zero(3);
            count = 0;
            int count1 = 0;
            VF = 0;
            for (int ii = i-1; ii <= i+1; ++ii) 
            {
              for (int jj = j-1; jj <= j+1; ++jj) 
              {
                for (int kk = k-1; kk <= k+1; ++kk) 
                {
                  if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                  {
                    if ((x-centered.row(count)).norm() <= 1.01)
                    {
                      double v = a_liquid_volume_fraction(ii,jj,kk);
                      mu(0) = mu(0) + centered.row(count)(0) * v;
                      mu(1) = mu(1) + centered.row(count)(1) * v;
                      mu(2) = mu(2) + centered.row(count)(2) * v;
                      VF = VF + v;
                      ++count1;
                    }
                    ++count;
                  }
                }
              }
            }
            if (VF > IRL::global_constants::VF_LOW)
            {
              mu = mu / VF;
            }
            // if (mus.size() > 0 && (mu - mus[mus.size()-1]).norm() < 0.001)
            // {
            //   flag = false;
            // }
            if (count1 <= 1)
            {
              flag = false;
            }
            else if (abs(x(0)) > 1.5 || abs(x(1)) > 1.5 || abs(x(2)) > 1.5)
            {
              flag = false;
            }
            else
            {
              if ((x - x_origin).norm() > 1e-5 && (mus.size() > 0 && (mu - mus[mus.size()-1]).norm() > 0.001))
              {
                mus.push_back(mu);
              }
              local.resize(count1, 3);
              count1 = 0;
              count = 0;
              for (int ii = i-1; ii <= i+1; ++ii) 
              {
                for (int jj = j-1; jj <= j+1; ++jj) 
                {
                  for (int kk = k-1; kk <= k+1; ++kk) 
                  {
                    if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW && a_liquid_volume_fraction(ii, jj, kk) < IRL::global_constants::VF_HIGH)
                    {
                      if ((x-centered.row(count)).norm() <= 1.01)
                      {
                        local.row(count1) = centered.row(count);
                        ++count1;
                      }
                      ++count;
                    }
                  }
                }
              }
              Eigen::MatrixXd cov = (local.transpose()*local) / (local.rows()-1);
              Eigen::EigenSolver<Eigen::MatrixXd> es(cov);
              Eigen::Index maxL;
              es.eigenvalues().real().maxCoeff(&maxL);
              Eigen::VectorXd dir_new = es.eigenvectors().real().col(maxL);
              dir_new.normalize();
              dir_new(0) = -dir_new(0);
              dir_new(1) = -dir_new(1);
              dir_new(2) = -dir_new(2);
              if (dir_new.dot(dir) < 0)
              {
                dir_new = -dir_new;
              }
              dir = dir_new;
              x(0) = x(0) + 1.0*dir(0);
              x(1) = x(1) + 1.0*dir(1);
              x(2) = x(2) + 1.0*dir(2);
            }
          }
          //std::cout << "hi " << mus.size() << std::endl;
          IRL::Normal direction = IRL::Normal(0.0,0.0,0.0);
          //for (int n = 0; n < mus.size()-1; ++n)
          {
            // std::cout << mus[n] << std::endl;
            // direction[0] = direction[0] + (mus[n+1](0) - mus[n](0));
            // direction[1] = direction[1] + (mus[n+1](1) - mus[n](1));
            // direction[2] = direction[2] + (mus[n+1](2) - mus[n](2));
          }
          //std::cout << mus[mus.size()-1] << std::endl;
          //std::cout << std::endl << std::endl;

          if (mus.size() <= 1)
          {
              Eigen::MatrixXd centered = bary.rowwise() - bary.colwise().mean(); 
              Eigen::MatrixXd cov = (centered.transpose()*centered) / (centered.rows()-1);
              Eigen::EigenSolver<Eigen::MatrixXd> es(cov);
              Eigen::Index maxL;
              es.eigenvalues().real().maxCoeff(&maxL);
              Eigen::VectorXd dir = es.eigenvectors().real().col(maxL);
              direction[0] = dir(0);
              direction[1] = dir(1);
              direction[2] = dir(2);
          }
          else
          {
            std::sort(mus.begin(),mus.end(),myfunction);
            for (int n = 0; n < mus.size()-1; ++n)
            {
              direction[0] = direction[0] + (mus[n+1](0) - mus[n](0));
              direction[1] = direction[1] + (mus[n+1](1) - mus[n](1));
              direction[2] = direction[2] + (mus[n+1](2) - mus[n](2));
            }
            // direction[0] = mus[mus.size()/2](0) - mus[mus.size()/2-1](0);
            // direction[1] = mus[mus.size()/2](1) - mus[mus.size()/2-1](1);
            // direction[2] = mus[mus.size()/2](2) - mus[mus.size()/2-1](2);
          }

          direction.normalize();
          //std::cout << direction << std::endl << std::endl << std::endl << std::endl;
          double n3 = 0;
          double n2 = 0;
          double n1 = 0;
          IRL::Normal v1;
          if (abs(direction[0]) >= abs(direction[1]) && abs(direction[0]) >= abs(direction[2]))
          {
            double n2 = direction[0]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n1 = (-n2*direction[1])/direction[0];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[1]) >= abs(direction[0]) && abs(direction[1]) >= abs(direction[2]))
          {
            double n1 = direction[1]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
            double n2 = (-n1*direction[0])/direction[1];
            v1[0] = n1; v1[1] = n2; v1[2] = 0;
          }
          else if (abs(direction[2]) >= abs(direction[0]) && abs(direction[2]) >= abs(direction[1]))
          {
            double n2 = direction[2]/(sqrt(direction[1]*direction[1]+direction[2]*direction[2]));
            double n3 = (-n2*direction[1])/direction[2];
            v1[0] = 0; v1[1] = n2; v1[2] = n3;
          }
          else
          {
            v1[0] = 0; v1[1] = 0; v1[2] = 0;
          }
          IRL::Normal b = IRL::crossProduct(direction,v1);
          b.normalize();
          IRL::Normal a = IRL::crossProduct(b,direction);
          a.normalize();
          IRL::ReferenceFrame frame = IRL::ReferenceFrame(direction, a, b);

          cylinder = IRL::Cylinder(datum, frame, 1, 0.00025);

          const IRL::Pt lower_cell_pt(mesh.x(i), mesh.y(j), mesh.z(k));
          const IRL::Pt upper_cell_pt(mesh.x(i + 1), mesh.y(j + 1),
                  mesh.z(k + 1));

          auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt,
                                          upper_cell_pt);
          IRL::ProgressiveRadiusSolverCylinder<IRL::RectangularCuboid>
          solver_radius(cell, a_liquid_volume_fraction(i, j, k), 1.0e-14,
          cylinder);

          cylinder = solver_radius.getCylinder();
          //std::cout << i << " " << j << " " << k << " " << (*a_interface)(i, j, k) << " " << cylinder << std::endl << std::endl << std::endl << std::endl;
          (*a_interface)(i, j, k) = cylinder;
        }
      }
    }
  }

  // Update border with simple ghost-cell fill and correct datum for
  // assumed periodic boundary
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}