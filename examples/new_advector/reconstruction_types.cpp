// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2019 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/new_advector/reconstruction_types.h"

#include <string.h>
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

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"
#include "examples/new_advector/vof_advection.h"
#include "examples/new_advector/plicnet.h"
#include "examples/new_advector/r2pnet.h"
#include "examples/new_advector/r2pnet_solve.h"
#include "examples/new_advector/ml_classifier.h"
#include "examples/new_advector/r2p_paraboloid_pass.h"

void getReconstruction(
    const std::string& a_reconstruction_method,
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  recon_method = Data<int>(&a_liquid_volume_fraction.getMesh());
  num_planes = Data<int>(&a_liquid_volume_fraction.getMesh());
  feature_class = Data<int>(&a_liquid_volume_fraction.getMesh());
  branch = Data<int>(&a_liquid_volume_fraction.getMesh());
  if (a_reconstruction_method == "ELVIRA2D") {
    ELVIRA2D::getReconstruction(a_liquid_volume_fraction, a_dt, a_U, a_V, a_W,
                                a_interface);
  } else if (a_reconstruction_method == "LVIRA2D") {
    LVIRA2D::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                               a_gas_centroid, a_dt, a_U, a_V, a_W,
                               a_interface);
  } else if (a_reconstruction_method == "MOF2D") {
    MOF2D::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                             a_gas_centroid, a_localized_separator_link, a_dt,
                             a_U, a_V, a_W, a_interface);
  } else if (a_reconstruction_method == "AdvectedNormals") {
    AdvectedNormals::getReconstruction(
        a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid,
        a_localized_separator_link, a_dt, a_U, a_V, a_W, a_interface);
  } else if (a_reconstruction_method == "R2P2D") {
    R2P2D::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                             a_gas_centroid, a_localized_separator_link, a_dt,
                             a_U, a_V, a_W, a_interface);
  } else if (a_reconstruction_method == "ELVIRA3D") {
    ELVIRA3D::getReconstruction(a_liquid_volume_fraction, a_dt, a_U, a_V, a_W,
                                a_interface);
  } else if (a_reconstruction_method == "LVIRA3D") {
    LVIRA3D::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                               a_gas_centroid, a_dt, a_U, a_V, a_W,
                               a_interface);
  } else if (a_reconstruction_method == "PLICNET") {
    PLICNET::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                               a_gas_centroid, a_dt, a_U, a_V, a_W,
                               a_interface);
  } else if (a_reconstruction_method == "MOF3D") {
    MOF3D::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                             a_gas_centroid, a_localized_separator_link, a_dt,
                             a_U, a_V, a_W, a_interface);
  } else if (a_reconstruction_method == "AdvectedNormals3D") {
    AdvectedNormals3D::getReconstruction(
        a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid,
        a_localized_separator_link, a_dt, a_U, a_V, a_W, a_interface);
  } else if (a_reconstruction_method == "R2P3D") {
    R2P3D::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                             a_gas_centroid, a_localized_separator_link, a_dt,
                             a_U, a_V, a_W, a_interface);
  } else if (a_reconstruction_method == "R2P3D_Hybrid") {
    R2P3D_Hybrid::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                             a_gas_centroid, a_localized_separator_link, a_dt,
                             a_U, a_V, a_W, a_interface);
  } else if (a_reconstruction_method == "R2P3D_Net") {
    R2P3D_Net::getReconstruction(a_liquid_volume_fraction, a_liquid_centroid,
                             a_gas_centroid, a_localized_separator_link, a_dt,
                             a_U, a_V, a_W, a_interface);
  } else {
    std::cout << "Unknown reconstruction method of : "
              << a_reconstruction_method << '\n';
    std::cout << "Value entries are: ELVIRA2D, LVIRA2D, MOF2D, "
                 "AdvectedNormals, R2P2D, ELVIRA3D, LVIRA3D, MOF3D, AdvectedNormals3D, R2P3D, R2P3D_Hybrid, R2P_Net. \n";
    std::exit(-1);
  }
}

void ELVIRA2D::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                 const double a_dt, const Data<double>& a_U,
                                 const Data<double>& a_V,
                                 const Data<double>& a_W,
                                 Data<IRL::PlanarSeparator>* a_interface) {
  IRL::ELVIRANeighborhood neighborhood;
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  neighborhood.resize(9);
  IRL::RectangularCuboid cells[9];
  // Loop over cells in domain. Skip if cell is not mixed phase.
  const int k = 0;
  const int kk = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
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
          // Reversed order, bad for cache locality but thats okay..
          cells[(jj - j + 1) * 3 + (ii - i + 1)] =
              IRL::RectangularCuboid::fromBoundingPts(
                  IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                  IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
          neighborhood.setMember(&cells[(jj - j + 1) * 3 + (ii - i + 1)],
                                 &a_liquid_volume_fraction(ii, jj, 0), ii - i,
                                 jj - j);
        }
      }
      // Now perform actual ELVIRA and obtain interface PlanarSeparator
      (*a_interface)(i, j, k) = reconstructionWithELVIRA2D(neighborhood);
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
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
  // const int k = 0;
  // const int kk = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }
        // Build surrounding stencil information for ELVIRA.
        for (int ii = i - 1; ii < i + 2; ++ii) {
          for (int jj = j - 1; jj < j + 2; ++jj) {
            for (int kk = k - 1; kk < k + 2; ++kk) {
              // Reversed order, bad for cache locality but thats okay..
              cells[9 * (kk - k + 1 ) + (jj - j + 1) * 3 + (ii - i + 1)] =
                  IRL::RectangularCuboid::fromBoundingPts(
                      IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                      IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
              neighborhood.setMember(&cells[9 * (kk - k + 1 ) + (jj - j + 1) * 3 + (ii - i + 1)],
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

void LVIRA2D::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const Data<IRL::Pt>& a_liquid_centroid,
                                const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface) {
  IRL::LVIRANeighborhood<IRL::RectangularCuboid> neighborhood;
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  neighborhood.resize(9);
  neighborhood.setCenterOfStencil(4);
  IRL::RectangularCuboid cells[9];
  // Loop over cells in domain. Skip if cell is not mixed phase.
  const int k = 0;
  const int kk = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
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
          // Reversed order, bad for cache locality but thats okay..
          cells[(jj - j + 1) * 3 + (ii - i + 1)] =
              IRL::RectangularCuboid::fromBoundingPts(
                  IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                  IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
          neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(
                                     (jj - j + 1) * 3 + (ii - i + 1)),
                                 &cells[(jj - j + 1) * 3 + (ii - i + 1)],
                                 &a_liquid_volume_fraction(ii, jj, kk));
        }
      }
      // Now create initial guess using centroids
      auto bary_normal = IRL::Normal::fromPtNormalized(
          a_gas_centroid(i, j, k) - a_liquid_centroid(i, j, k));
      bary_normal[2] = 0.0;
      bary_normal.normalize();
      const double initial_distance =
          bary_normal * neighborhood.getCenterCell().calculateCentroid();
      (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
          IRL::Plane(bary_normal, initial_distance));
      setDistanceToMatchVolumeFractionPartialFill(
          neighborhood.getCenterCell(),
          neighborhood.getCenterCellStoredMoments(), &(*a_interface)(i, j, k));

      (*a_interface)(i, j, k) =
          reconstructionWithLVIRA2D(neighborhood, (*a_interface)(i, j, k));
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void LVIRA3D::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const Data<IRL::Pt>& a_liquid_centroid,
                                const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface) {
  IRL::LVIRANeighborhood<IRL::RectangularCuboid> neighborhood;
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  neighborhood.resize(27);
  neighborhood.setCenterOfStencil(13);
  IRL::RectangularCuboid cells[27];
  // Loop over cells in domain. Skip if cell is not mixed phase.
  // const int k = 0;
  // const int kk = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) >
                IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }
        // Build surrounding stencil information for ELVIRA.
        for (int ii = i - 1; ii < i + 2; ++ii) {
          for (int jj = j - 1; jj < j + 2; ++jj) {
            for (int kk = k - 1; kk < k + 2; ++kk) {
              // Reversed order, bad for cache locality but thats okay..
              const int local_index =
                  (kk - k + 1) * 9 + (jj - j + 1) * 3 + (ii - i + 1);
              cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
                  IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                  IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
              neighborhood.setMember(
                  static_cast<IRL::UnsignedIndex_t>(local_index),
                  &cells[local_index], &a_liquid_volume_fraction(ii, jj, kk));
            }
          }
        }
        // Now create initial guess using centroids
        auto bary_normal = IRL::Normal::fromPtNormalized(
            a_gas_centroid(i, j, k) - a_liquid_centroid(i, j, k));
        bary_normal.normalize();
        const double initial_distance =
            bary_normal * neighborhood.getCenterCell().calculateCentroid();
        (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
            IRL::Plane(bary_normal, initial_distance));
        setDistanceToMatchVolumeFractionPartialFill(
            neighborhood.getCenterCell(),
            neighborhood.getCenterCellStoredMoments(),
            &(*a_interface)(i, j, k));

        (*a_interface)(i, j, k) =
            reconstructionWithLVIRA3D(neighborhood, (*a_interface)(i, j, k));
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void PLICNET::getReconstruction(const Data<double>& a_liquid_volume_fraction,
                                const Data<IRL::Pt>& a_liquid_centroid,
                                const Data<IRL::Pt>& a_gas_centroid,
                                const double a_dt, const Data<double>& a_U,
                                const Data<double>& a_V,
                                const Data<double>& a_W,
                                Data<IRL::PlanarSeparator>* a_interface) {
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  double moments[189] = {0};
  bool flip = false;
  double m000 = 0;
  double m100 = 0;
  double m010 = 0;
  double m001 = 0;
  double center[3] = {0};
  int direction = 0;
  int direction2 = 0;
  double n[3] = {0};
  IRL::Normal normal;
  double temp = 0;
  // Loop over cells in domain. Skip if cell is not mixed phase.
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) >
                IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }
        // Liquid-gas symmetry
        flip = false;
        if (a_liquid_volume_fraction(i, j, k) >= 0.5) flip = true;
        memset(moments, 0, sizeof(moments)); 
        m000 = 0; 
        m100 = 0; 
        m010 = 0; 
        m001 = 0;
        // Build surrounding stencil information for PLICNET.
        if (flip)
        {
          for (int ii = i - 1; ii < i + 2; ++ii) 
          {
            for (int jj = j - 1; jj < j + 2; ++jj) 
            {
              for (int kk = k - 1; kk < k + 2; ++kk) 
              {
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]=1.0-a_liquid_volume_fraction(ii, jj, kk);
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+1]=(a_gas_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+2]=(a_gas_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+3]=(a_gas_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+4]=(a_liquid_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+5]=(a_liquid_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+6]=(a_liquid_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                // Calculate geometric moments of neighborhood
                m000=m000+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
                m100=m100+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+1]+(ii-i))*(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
                m010=m010+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+2]+(jj-j))*(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
                m001=m001+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+3]+(kk-k))*(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
              }
            }
          }
        }
        else
        { 
          for (int ii = i - 1; ii < i + 2; ++ii) 
          {
            for (int jj = j - 1; jj < j + 2; ++jj) 
            {
              for (int kk = k - 1; kk < k + 2; ++kk) 
              {
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]=a_liquid_volume_fraction(ii, jj, kk);
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+1]=(a_liquid_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+2]=(a_liquid_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+3]=(a_liquid_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+4]=(a_gas_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+5]=(a_gas_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+6]=(a_gas_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                // Calculate geometric moments of neighborhood
                m000=m000+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
                m100=m100+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+1]+(ii-i))*(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
                m010=m010+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+2]+(jj-j))*(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
                m001=m001+(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))+3]+(kk-k))*(moments[7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k))]);
              }
            }
          }
        }
        // Calculate geometric center of neighborhood
        center[0] = m100/m000;
        center[1] = m010/m000;
        center[2] = m001/m000;
        // Symmetry about Cartesian planes
        direction = 0;
        direction2 = 0;
        plicnet::reflect_moments(moments,center,&direction,&direction2);
        // Get PLIC normal vector from neural network
        n[0] = 0;
        n[1] = 0;
        n[2] = 0;
        plicnet::get_normal(moments, n);
        normal = IRL::Normal(n[0],n[1],n[2]);
        // Rotate normal vector to original octant
        switch (direction2)
        {
          case 1:
            temp=normal[0];
            normal[0]=normal[1];
            normal[1]=temp;
            break;
          case 2:
            temp=normal[1];
            normal[1]=normal[2];
            normal[2]=temp;
            break;
          case 3:
            temp=normal[0];
            normal[0]=normal[2];
            normal[2]=temp;
            break;
          case 4:
            temp=normal[1];
            normal[1]=normal[2];
            normal[2]=temp;
            temp=normal[0];
            normal[0]=normal[1];
            normal[1]=temp;
            break;
          case 5:
            temp=normal[0];
            normal[0]=normal[2];
            normal[2]=temp;
            temp=normal[0];
            normal[0]=normal[1];
            normal[1]=temp;
            break;
        }

        switch (direction)
        {
          case 1:
            normal[0] = -normal[0];
            break;
          case 2:
            normal[1] = -normal[1];
            break;
          case 3:
            normal[2] = -normal[2];
            break;
          case 4:
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            break;
          case 5:
            normal[0] = -normal[0];
            normal[2] = -normal[2];
            break;
          case 6:
            normal[1] = -normal[1];
            normal[2] = -normal[2];
            break;
          case 7:
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
            break;
        }

        if (!flip)
        {
          normal[0] = -normal[0];
          normal[1] = -normal[1];
          normal[2] = -normal[2];
        }

        normal[0] = normal[0] * mesh.dx();
        normal[1] = normal[1] * mesh.dy();
        normal[2] = normal[2] * mesh.dz();
        normal.normalize();
        
        const IRL::Normal& n1 = normal;
        const double vf = a_liquid_volume_fraction(i,j,k);
        const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        double distance = IRL::findDistanceOnePlane(cube, vf, n1);
        (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal, distance));
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void MOF2D::getReconstruction(
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();

  const int k = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
          a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
        const double distance =
            std::copysign(IRL::global_constants::ARBITRARILY_LARGE_DISTANCE,
                          a_liquid_volume_fraction(i, j, k) - 0.5);
        (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
            IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
        continue;
      }
      auto cell = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
          IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
      double vol = cell.calculateVolume();
      IRL::SeparatedMoments<IRL::VolumeMoments> svm(
          IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                             a_liquid_centroid(i, j, k)),
          IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                             a_gas_centroid(i, j, k)));
      (*a_interface)(i, j, k) =
          IRL::reconstructionWithMOF2D(cell, svm, 0.5, 0.5);
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void MOF3D::getReconstruction(
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();

  // const int k = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        double vol = cell.calculateVolume();
        IRL::SeparatedMoments<IRL::VolumeMoments> svm(
            IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                              a_liquid_centroid(i, j, k)),
            IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                              a_gas_centroid(i, j, k)));
        (*a_interface)(i, j, k) =
            IRL::reconstructionWithMOF3D(cell, svm, 0.5, 0.5);
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void AdvectedNormals::getReconstruction(
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  // Get mesh everything is living on.
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  // Container for moments from advection
  Data<IRL::ListedVolumeMoments<IRL::VolumeMomentsAndNormal>> listed_moments(
      &mesh);

  const int k = 0;
  const int kk = 0;
  for (int i = mesh.imino() + 1; i <= mesh.imaxo() - 1; ++i) {
    for (int j = mesh.jmino() + 1; j <= mesh.jmaxo() - 1; ++j) {
      auto cell = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
          IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
      const auto localizer_link = IRL::LocalizerLinkFromLocalizedSeparatorLink(
          &a_localized_separator_link(i, j, k));
      for (IRL::UnsignedIndex_t n = 0;
           n < (*a_interface)(i, j, k).getNumberOfPlanes(); ++n) {
        IRL::Polygon interface_poly =
            IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                cell, (*a_interface)(i, j, k), (*a_interface)(i, j, k)[n]);
        if (interface_poly.getNumberOfVertices() == 0) {
          continue;
        }
        for (IRL::UnsignedIndex_t tri = 0;
             tri < interface_poly.getNumberOfSimplicesInDecomposition();
             ++tri) {
          IRL::Tri simplex = static_cast<IRL::Tri>(
              interface_poly.getSimplexFromDecomposition(tri));
          for (auto& vertex : simplex) {
            vertex = back_project_vertex(vertex, a_dt, a_U, a_V, a_W);
          }
          simplex.calculateAndSetPlaneOfExistence();
          auto new_moments =
              IRL::getVolumeMoments<IRL::TaggedAccumulatedListedVolumeMoments<
                  IRL::VolumeMomentsAndNormal>>(simplex, localizer_link);
          for (IRL::UnsignedIndex_t moment = 0; moment < new_moments.size();
               ++moment) {
            auto index_for_tag =
                getIndexFromTag(mesh, new_moments.getTagForIndex(moment));
            listed_moments(index_for_tag[0], index_for_tag[1],
                           index_for_tag[2]) +=
                new_moments.getMomentsForIndex(moment);
          }
        }
      }
    }
  }

  // Remove Z components from advected surface elements that will be used
  // This can occur by polygons being rotated by the flow field during
  // advection.
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      const IRL::UnsignedIndex_t starting_length =
          listed_moments(i, j, k).size();
      for (IRL::UnsignedIndex_t n = starting_length - 1;
           n != static_cast<IRL::UnsignedIndex_t>(-1); --n) {
        IRL::VolumeMomentsAndNormal& moment = listed_moments(i, j, k)[n];
        moment.normalizeByVolume();
        moment.normal()[2] = 0.0;
        moment.normal().normalize();
        if (moment.normal().calculateMagnitude() < 0.95) {
          listed_moments(i, j, k).erase(n);
        } else {
          moment.multiplyByVolume();
        }
      }
    }
  }

  // Now have all of the advected moments. Get and store the reconstructions
  // from this by partitioning with Kmeans.
  IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
  ////////////////////////////////////////////////
  neighborhood.resize(9);  // SET TO 9 BECAUSE OF 2D.
  neighborhood.setCenterOfStencil(4);
  IRL::RectangularCuboid stencil_cells[9];
  IRL::SeparatedMoments<IRL::VolumeMoments> stencil_moments[9];
  int num_mof = 0;
  int num_adv = 0;
  int num_adv2 = 0;
  ////////////////////////////////////////////////
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
          a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
        const double distance =
            std::copysign(IRL::global_constants::ARBITRARILY_LARGE_DISTANCE,
                          a_liquid_volume_fraction(i, j, k) - 0.5);
        (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
            IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
        continue;
      } else if (listed_moments(i, j, k).size() == 0) {
        // No interface advected in, use MoF
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        double vol = cell.calculateVolume();
        IRL::SeparatedMoments<IRL::VolumeMoments> svm(
            IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                               a_liquid_centroid(i, j, k)),
            IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                               a_gas_centroid(i, j, k)));
        (*a_interface)(i, j, k) = IRL::reconstructionWithMOF2D(cell, svm);
        ++num_mof;
      } else {
        // Set up R2P neighborhood
        for (int ii = i - 1; ii < i + 2; ++ii) {
          for (int jj = j - 1; jj < j + 2; ++jj) {
            const int ind = (ii - i + 1) * 3 + (jj - j + 1);
            stencil_cells[ind] = IRL::RectangularCuboid::fromBoundingPts(
                IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(k)),
                IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(k + 1)));
            double vol = stencil_cells[ind].calculateVolume();
            stencil_moments[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
                IRL::VolumeMoments(a_liquid_volume_fraction(ii, jj, kk) * vol,
                                   a_liquid_centroid(ii, jj, kk)),
                IRL::VolumeMoments(
                    (1.0 - a_liquid_volume_fraction(ii, jj, kk)) * vol,
                    a_gas_centroid(ii, jj, kk)));
            neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind),
                                   &stencil_cells[ind], &stencil_moments[ind]);
          }
        }
        (*a_interface)(i, j, k) = IRL::reconstructionWithAdvectedNormals(
            listed_moments(i, j, k), neighborhood);
        ++num_adv;
        if ((*a_interface)(i, j, k).getNumberOfPlanes() == 2) {
          ++num_adv2;
        }
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void AdvectedNormals3D::getReconstruction(
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  // Get mesh everything is living on.
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  // Container for moments from advection
  Data<IRL::ListedVolumeMoments<IRL::VolumeMomentsAndNormal>> listed_moments(
      &mesh);

  // const int k = 0;
  // const int kk = 0;
  for (int i = mesh.imino() + 1; i <= mesh.imaxo() - 1; ++i) {
    for (int j = mesh.jmino() + 1; j <= mesh.jmaxo() - 1; ++j) {
      for (int k = mesh.kmino() + 1; k <= mesh.kmaxo() - 1; ++k) {
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        const auto localizer_link = IRL::LocalizerLinkFromLocalizedSeparatorLink(
            &a_localized_separator_link(i, j, k));
        for (IRL::UnsignedIndex_t n = 0;
            n < (*a_interface)(i, j, k).getNumberOfPlanes(); ++n) {
          IRL::Polygon interface_poly =
              IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                  cell, (*a_interface)(i, j, k), (*a_interface)(i, j, k)[n]);
          if (interface_poly.getNumberOfVertices() == 0) {
            continue;
          }
          for (IRL::UnsignedIndex_t tri = 0;
              tri < interface_poly.getNumberOfSimplicesInDecomposition();
              ++tri) {
            IRL::Tri simplex = static_cast<IRL::Tri>(
                interface_poly.getSimplexFromDecomposition(tri));
            for (auto& vertex : simplex) {
              vertex = back_project_vertex(vertex, a_dt, a_U, a_V, a_W);
            }
            simplex.calculateAndSetPlaneOfExistence();
            auto new_moments =
                IRL::getVolumeMoments<IRL::TaggedAccumulatedListedVolumeMoments<
                    IRL::VolumeMomentsAndNormal>>(simplex, localizer_link);
            for (IRL::UnsignedIndex_t moment = 0; moment < new_moments.size();
                ++moment) {
              auto index_for_tag =
                  getIndexFromTag(mesh, new_moments.getTagForIndex(moment));
              listed_moments(index_for_tag[0], index_for_tag[1],
                            index_for_tag[2]) +=
                  new_moments.getMomentsForIndex(moment);
            }
          }
        }
      }
    }
  }

  // Remove Z components from advected surface elements that will be used
  // This can occur by polygons being rotated by the flow field during
  // advection.
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        const IRL::UnsignedIndex_t starting_length =
            listed_moments(i, j, k).size();
        for (IRL::UnsignedIndex_t n = starting_length - 1;
            n != static_cast<IRL::UnsignedIndex_t>(-1); --n) {
          IRL::VolumeMomentsAndNormal& moment = listed_moments(i, j, k)[n];
          moment.normalizeByVolume();
          // moment.normal()[2] = 0.0;
          moment.normal().normalize();
          if (moment.normal().calculateMagnitude() < 0.95) {
            listed_moments(i, j, k).erase(n);
          } else {
            moment.multiplyByVolume();
          }
        }
      }
    }
  }

  // Now have all of the advected moments. Get and store the reconstructions
  // from this by partitioning with Kmeans.
  IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
  ////////////////////////////////////////////////
  neighborhood.resize(27);  // SET TO 9 BECAUSE OF 2D.
  neighborhood.setCenterOfStencil(13);
  IRL::RectangularCuboid stencil_cells[27];
  IRL::SeparatedMoments<IRL::VolumeMoments> stencil_moments[27];
  int num_mof = 0;
  int num_adv = 0;
  int num_adv2 = 0;
  ////////////////////////////////////////////////
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance =
              std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        } else if (listed_moments(i, j, k).size() == 0) {
          // No interface advected in, use MoF
          auto cell = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
              IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
          double vol = cell.calculateVolume();
          IRL::SeparatedMoments<IRL::VolumeMoments> svm(
              IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                                a_liquid_centroid(i, j, k)),
              IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                                a_gas_centroid(i, j, k)));
          (*a_interface)(i, j, k) = IRL::reconstructionWithMOF3D(cell, svm);
          ++num_mof;
        } else {
          // Set up R2P neighborhood
          for (int ii = i - 1; ii < i + 2; ++ii) {
            for (int jj = j - 1; jj < j + 2; ++jj) {
              for (int kk = k - 1; kk < k + 2; ++kk) {
                const int ind = (ii - i + 1) * 9 + (jj - j + 1) * 3 + (kk - k + 1);
                stencil_cells[ind] = IRL::RectangularCuboid::fromBoundingPts(
                    IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                    IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
                double vol = stencil_cells[ind].calculateVolume();
                stencil_moments[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
                    IRL::VolumeMoments(a_liquid_volume_fraction(ii, jj, kk) * vol,
                                      a_liquid_centroid(ii, jj, kk)),
                    IRL::VolumeMoments(
                        (1.0 - a_liquid_volume_fraction(ii, jj, kk)) * vol,
                        a_gas_centroid(ii, jj, kk)));
                neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind),
                                      &stencil_cells[ind], &stencil_moments[ind]);
              }
            }
          }
          (*a_interface)(i, j, k) = IRL::reconstructionWithAdvectedNormals(
              listed_moments(i, j, k), neighborhood);
          ++num_adv;
          if ((*a_interface)(i, j, k).getNumberOfPlanes() == 2) {
            ++num_adv2;
          }
        }
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}

void R2P2D::getReconstruction(
    const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  // Get mesh everything is living on.
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  // Container for moments from advection
  Data<IRL::ListedVolumeMoments<IRL::VolumeMomentsAndNormal>> listed_moments(
      &mesh);

  const int k = 0;
  const int kk = 0;
  for (int i = mesh.imino() + 1; i <= mesh.imaxo() - 1; ++i) {
    for (int j = mesh.jmino() + 1; j <= mesh.jmaxo() - 1; ++j) {
      auto cell = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
          IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
      const auto localizer_link = IRL::LocalizerLinkFromLocalizedSeparatorLink(
          &a_localized_separator_link(i, j, k));
      for (IRL::UnsignedIndex_t n = 0;
           n < (*a_interface)(i, j, k).getNumberOfPlanes(); ++n) {
        IRL::Polygon interface_poly =
            IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                cell, (*a_interface)(i, j, k), (*a_interface)(i, j, k)[n]);
        if (interface_poly.getNumberOfVertices() == 0) {
          continue;
        }
        for (IRL::UnsignedIndex_t tri = 0;
             tri < interface_poly.getNumberOfSimplicesInDecomposition();
             ++tri) {
          IRL::Tri simplex = static_cast<IRL::Tri>(
              interface_poly.getSimplexFromDecomposition(tri));
          for (auto& vertex : simplex) {
            vertex = back_project_vertex(vertex, a_dt, a_U, a_V, a_W);
          }
          simplex.calculateAndSetPlaneOfExistence();
          auto new_moments =
              IRL::getVolumeMoments<IRL::TaggedAccumulatedListedVolumeMoments<
                  IRL::VolumeMomentsAndNormal>>(simplex, localizer_link);
          for (IRL::UnsignedIndex_t moment = 0; moment < new_moments.size();
               ++moment) {
            auto index_for_tag =
                getIndexFromTag(mesh, new_moments.getTagForIndex(moment));
            listed_moments(index_for_tag[0], index_for_tag[1],
                           index_for_tag[2]) +=
                new_moments.getMomentsForIndex(moment);
          }
        }
      }
    }
  }

  // Remove Z components from advected surface elements that will be used
  // This can occur by polygons being rotated by the flow field during
  // advection.
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      const IRL::UnsignedIndex_t starting_length =
          listed_moments(i, j, k).size();
      for (IRL::UnsignedIndex_t n = starting_length - 1;
           n != static_cast<IRL::UnsignedIndex_t>(-1); --n) {
        IRL::VolumeMomentsAndNormal& moment = listed_moments(i, j, k)[n];
        moment.normalizeByVolume();
        moment.normal()[2] = 0.0;
        moment.normal().normalize();
        if (moment.normal().calculateMagnitude() < 0.95) {
          listed_moments(i, j, k).erase(n);
        } else {
          moment.multiplyByVolume();
        }
      }
    }
  }

  // Now have all of the advected moments. Get and store the reconstructions
  // from this by partitioning with Kmeans.
  IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
  ////////////////////////////////////////////////
  neighborhood.resize(9);  // SET TO 9 BECAUSE OF 2D.
  neighborhood.setCenterOfStencil(4);
  IRL::RectangularCuboid stencil_cells[9];
  IRL::SeparatedMoments<IRL::VolumeMoments> stencil_moments[9];
  int num_mof = 0;
  int num_adv = 0;
  int num_adv2 = 0;
  ////////////////////////////////////////////////
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
          a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
        const double distance =
            std::copysign(IRL::global_constants::ARBITRARILY_LARGE_DISTANCE,
                          a_liquid_volume_fraction(i, j, k) - 0.5);
        (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
            IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
        continue;
      }

      // Set up R2P neighborhood
      for (int ii = i - 1; ii < i + 2; ++ii) {
        for (int jj = j - 1; jj < j + 2; ++jj) {
          const int ind = (ii - i + 1) * 3 + (jj - j + 1);
          stencil_cells[ind] = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(k)),
              IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(k + 1)));
          double vol = stencil_cells[ind].calculateVolume();
          stencil_moments[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
              IRL::VolumeMoments(a_liquid_volume_fraction(ii, jj, kk) * vol,
                                 a_liquid_centroid(ii, jj, kk)),
              IRL::VolumeMoments(
                  (1.0 - a_liquid_volume_fraction(ii, jj, kk)) * vol,
                  a_gas_centroid(ii, jj, kk)));
          neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind),
                                 &stencil_cells[ind], &stencil_moments[ind]);
        }
      }

      if (listed_moments(i, j, k).size() == 0) {
        // No interface advected in, use MoF
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        double vol = cell.calculateVolume();
        IRL::SeparatedMoments<IRL::VolumeMoments> svm(
            IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                               a_liquid_centroid(i, j, k)),
            IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                               a_gas_centroid(i, j, k)));
        (*a_interface)(i, j, k) = IRL::reconstructionWithMOF2D(cell, svm);
        ++num_mof;
        neighborhood.setSurfaceArea(
            getReconstructionSurfaceArea(cell, (*a_interface)(i, j, k)));
      } else {
        (*a_interface)(i, j, k) = IRL::reconstructionWithAdvectedNormals(
            listed_moments(i, j, k), neighborhood);
        ++num_adv;
        if ((*a_interface)(i, j, k).getNumberOfPlanes() == 2) {
          ++num_adv2;
        }
        double area_sum = 0.0;
        for (const auto& moment : listed_moments(i, j, k)) {
          area_sum += moment.volumeMoments().volume();
        }
        neighborhood.setSurfaceArea(area_sum);
      }
      (*a_interface)(i, j, k) =
          reconstructionWithR2P2D(neighborhood, (*a_interface)(i, j, k));
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}
























void R2P3D::getReconstruction(
const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  
  Data<IRL::ListedVolumeMoments<IRL::VolumeMomentsAndNormal>> listed_moments(&mesh);

  for (int i = mesh.imino() + 1; i <= mesh.imaxo() - 1; ++i) {
    for (int j = mesh.jmino() + 1; j <= mesh.jmaxo() - 1; ++j) {
      for (int k = mesh.kmino() + 1; k <= mesh.kmaxo() - 1; ++k) {
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        const auto localizer_link = IRL::LocalizerLinkFromLocalizedSeparatorLink(
            &a_localized_separator_link(i, j, k));
        for (IRL::UnsignedIndex_t n = 0; n < (*a_interface)(i, j, k).getNumberOfPlanes(); ++n) {
          IRL::Polygon interface_poly =
              IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                  cell, (*a_interface)(i, j, k), (*a_interface)(i, j, k)[n]);
          if (interface_poly.getNumberOfVertices() == 0) continue;
          
          for (IRL::UnsignedIndex_t tri = 0; tri < interface_poly.getNumberOfSimplicesInDecomposition(); ++tri) {
            IRL::Tri simplex = static_cast<IRL::Tri>(interface_poly.getSimplexFromDecomposition(tri));
            for (auto& vertex : simplex) {
              vertex = back_project_vertex(vertex, a_dt, a_U, a_V, a_W);
            }
            simplex.calculateAndSetPlaneOfExistence();
            auto new_moments =
                IRL::getVolumeMoments<IRL::TaggedAccumulatedListedVolumeMoments<
                    IRL::VolumeMomentsAndNormal>>(simplex, localizer_link);
            for (IRL::UnsignedIndex_t moment = 0; moment < new_moments.size(); ++moment) {
              auto index_for_tag = getIndexFromTag(mesh, new_moments.getTagForIndex(moment));
              listed_moments(index_for_tag[0], index_for_tag[1], index_for_tag[2]) +=
                  new_moments.getMomentsForIndex(moment);
            }
          }
        }
      }
    }
  }

  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        const IRL::UnsignedIndex_t starting_length = listed_moments(i, j, k).size();
        for (IRL::UnsignedIndex_t n = starting_length - 1; n != static_cast<IRL::UnsignedIndex_t>(-1); --n) {
          IRL::VolumeMomentsAndNormal& moment = listed_moments(i, j, k)[n];
          moment.normalizeByVolume();
          moment.normal().normalize();
          if (moment.normal().calculateMagnitude() < 0.95) {
            listed_moments(i, j, k).erase(n);
          } else {
            moment.multiplyByVolume();
          }
        }
      }
    }
  }

  IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
  neighborhood.resize(27);
  neighborhood.setCenterOfStencil(13);
  IRL::RectangularCuboid stencil_cells[27];
  IRL::SeparatedMoments<IRL::VolumeMoments> stencil_moments[27];

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance = std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }

        recon_method(i, j, k) = 1;
        // --- R2P3D Logic ---
        for (int ii = i - 1; ii < i + 2; ++ii) {
          for (int jj = j - 1; jj < j + 2; ++jj) {
            for (int kk = k - 1; kk < k + 2; ++kk) {
              const int ind = (ii - i + 1) * 9 + (jj - j + 1) * 3 + (kk - k + 1);
              stencil_cells[ind] = IRL::RectangularCuboid::fromBoundingPts(
                  IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                  IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
              double vol = stencil_cells[ind].calculateVolume();
              stencil_moments[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
                  IRL::VolumeMoments(a_liquid_volume_fraction(ii, jj, kk) * vol,
                                      a_liquid_centroid(ii, jj, kk)),
                  IRL::VolumeMoments(
                      (1.0 - a_liquid_volume_fraction(ii, jj, kk)) * vol,
                      a_gas_centroid(ii, jj, kk)));
              neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind),
                                      &stencil_cells[ind],
                                      &stencil_moments[ind]);
            }
          }
        }

        if (listed_moments(i, j, k).size() == 0) {
          // No advected interface, use MOF3D
          auto cell = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
              IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
          double vol = cell.calculateVolume();
          IRL::SeparatedMoments<IRL::VolumeMoments> svm(
              IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                                  a_liquid_centroid(i, j, k)),
              IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                                  a_gas_centroid(i, j, k)));
          (*a_interface)(i, j, k) = IRL::reconstructionWithMOF3D(cell, svm);
          neighborhood.setSurfaceArea(getReconstructionSurfaceArea(cell, (*a_interface)(i, j, k)));
        } else {
          // Use AdvectedNormals
          (*a_interface)(i, j, k) = IRL::reconstructionWithAdvectedNormals(listed_moments(i, j, k), neighborhood);
          double area_sum = 0.0;
          for (const auto& moment : listed_moments(i, j, k)) {
            area_sum += moment.volumeMoments().volume();
          }
          neighborhood.setSurfaceArea(area_sum);
        }

        (*a_interface)(i, j, k) = reconstructionWithR2P3D(neighborhood, (*a_interface)(i, j, k));

        if ((*a_interface)(i, j, k).getNumberOfPlanes() == 1)
        {
          num_planes(i, j, k) = 1;
        }
        else if ((*a_interface)(i, j, k).getNumberOfPlanes() == 2)
        {
          num_planes(i, j, k) = 2;
        }
        else
        {
          num_planes(i, j, k) = 0;
        }
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}























void R2P3D_Hybrid::getReconstruction(
const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  
  // Get mesh everything is living on.
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  
  // Container for moments from advection
  Data<IRL::ListedVolumeMoments<IRL::VolumeMomentsAndNormal>> listed_moments(&mesh);

  // 1. Back-Project Surface Elements
  for (int i = mesh.imino() + 1; i <= mesh.imaxo() - 1; ++i) {
    for (int j = mesh.jmino() + 1; j <= mesh.jmaxo() - 1; ++j) {
      for (int k = mesh.kmino() + 1; k <= mesh.kmaxo() - 1; ++k) {
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        const auto localizer_link = IRL::LocalizerLinkFromLocalizedSeparatorLink(
            &a_localized_separator_link(i, j, k));
        for (IRL::UnsignedIndex_t n = 0; n < (*a_interface)(i, j, k).getNumberOfPlanes(); ++n) {
          IRL::Polygon interface_poly =
              IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                  cell, (*a_interface)(i, j, k), (*a_interface)(i, j, k)[n]);
          if (interface_poly.getNumberOfVertices() == 0) continue;
          
          for (IRL::UnsignedIndex_t tri = 0; tri < interface_poly.getNumberOfSimplicesInDecomposition(); ++tri) {
            IRL::Tri simplex = static_cast<IRL::Tri>(interface_poly.getSimplexFromDecomposition(tri));
            for (auto& vertex : simplex) {
              vertex = back_project_vertex(vertex, a_dt, a_U, a_V, a_W);
            }
            simplex.calculateAndSetPlaneOfExistence();
            auto new_moments =
                IRL::getVolumeMoments<IRL::TaggedAccumulatedListedVolumeMoments<
                    IRL::VolumeMomentsAndNormal>>(simplex, localizer_link);
            for (IRL::UnsignedIndex_t moment = 0; moment < new_moments.size(); ++moment) {
              auto index_for_tag = getIndexFromTag(mesh, new_moments.getTagForIndex(moment));
              listed_moments(index_for_tag[0], index_for_tag[1], index_for_tag[2]) +=
                  new_moments.getMomentsForIndex(moment);
            }
          }
        }
      }
    }
  }

  // 2. Clean Up Advected Elements
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        const IRL::UnsignedIndex_t starting_length = listed_moments(i, j, k).size();
        for (IRL::UnsignedIndex_t n = starting_length - 1; n != static_cast<IRL::UnsignedIndex_t>(-1); --n) {
          IRL::VolumeMomentsAndNormal& moment = listed_moments(i, j, k)[n];
          moment.normalizeByVolume();
          moment.normal().normalize();
          if (moment.normal().calculateMagnitude() < 0.95) {
            listed_moments(i, j, k).erase(n);
          } else {
            moment.multiplyByVolume();
          }
        }
      }
    }
  }

  // 3. Zonghao's Colinearity Metric Allocation
  Data<double> norm_pos(&mesh);
  Data<double> norm_neg(&mesh);
  Data<double> tmp_pos(&mesh);
  Data<double> tmp_neg(&mesh);

  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        norm_pos(i, j, k) = 0.0;
        norm_neg(i, j, k) = 0.0;
        tmp_pos(i, j, k) = 0.0;
        tmp_neg(i, j, k) = 0.0;
      }
    }
  }

  // 4. Compute Colinearity Metric
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          continue;
        }
        
        double surface_area = 0.0;
        std::vector<IRL::Normal> normals_adj;
        std::vector<double> area_adj;

        for (int ii = i - 1; ii <= i + 1; ++ii) {
          for (int jj = j - 1; jj <= j + 1; ++jj) {
            for (int kk = k - 1; kk <= k + 1; ++kk) {
              for (IRL::UnsignedIndex_t ind = 0; ind < listed_moments(ii, jj, kk).size(); ++ind) {
                IRL::VolumeMomentsAndNormal moment = listed_moments(ii, jj, kk)[ind];
                double area = moment.volumeMoments().volume();
                IRL::Normal n_adj = moment.normal();
                n_adj.normalize();
                normals_adj.push_back(n_adj);
                area_adj.push_back(area);
                surface_area += area;
              }
            }
          }
        }

        if (surface_area > 0.0) {
          double surf_dot_pos_sum = 0.0;
          double surf_dot_neg_sum = 0.0;
          int size_adj = normals_adj.size();
          std::vector<double> norm_pos_loc(size_adj, 0.0);
          std::vector<double> norm_neg_loc(size_adj, 0.0);

          for (int n = 0; n < size_adj; ++n) {
            for (int nn = 0; nn < size_adj; ++nn) {
              if (n == nn) continue;
              double dot_result = normals_adj[n] * normals_adj[nn];
              if (dot_result >= 0.0) norm_pos_loc[n] += area_adj[nn] * dot_result;
              if (dot_result < 0.0)  norm_neg_loc[n] -= area_adj[nn] * dot_result;
            }
            norm_pos_loc[n] /= (surface_area - area_adj[n]);
            norm_neg_loc[n] /= (surface_area - area_adj[n]);
          }

          for (int n = 0; n < size_adj; ++n) {
            surf_dot_pos_sum += norm_pos_loc[n] * area_adj[n];
            surf_dot_neg_sum += norm_neg_loc[n] * area_adj[n];
          }
          norm_pos(i, j, k) = surf_dot_pos_sum / surface_area;
          norm_neg(i, j, k) = surf_dot_neg_sum / surface_area;
        }
      }
    }
  }

  // 5. Filter Metric
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          continue;
        }

        double surface_area = 0.0;
        for (int ii = i - 1; ii <= i + 1; ++ii) {
          for (int jj = j - 1; jj <= j + 1; ++jj) {
            for (int kk = k - 1; kk <= k + 1; ++kk) {
              double cell_area = 0.0;
              for (IRL::UnsignedIndex_t ind = 0; ind < listed_moments(ii, jj, kk).size(); ++ind) {
                cell_area += listed_moments(ii, jj, kk)[ind].volumeMoments().volume();
              }
              surface_area += cell_area;
              tmp_pos(i, j, k) += cell_area * norm_pos(ii, jj, kk);
              tmp_neg(i, j, k) += cell_area * norm_neg(ii, jj, kk);
            }
          }
        }
        if (surface_area > 0.0) {
          tmp_pos(i, j, k) /= surface_area;
          tmp_neg(i, j, k) /= surface_area;
        }
      }
    }
  }

  // Assign Filtered Metrics
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        norm_pos(i, j, k) = tmp_pos(i, j, k);
        norm_neg(i, j, k) = tmp_neg(i, j, k);
      }
    }
  }

  // 6. Hybrid R2PNET Main Evaluation and Reconstruction Loop
  IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
  neighborhood.resize(27);
  neighborhood.setCenterOfStencil(13);
  IRL::RectangularCuboid stencil_cells[27];
  IRL::SeparatedMoments<IRL::VolumeMoments> stencil_moments[27];

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance = std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }

        double n_pos = norm_pos(i, j, k);
        double n_neg = norm_neg(i, j, k);

        double vol = 0;
        for (int ii=i-1; ii<i+2; ++ii)
          for (int jj=j-1; jj<j+2; ++jj)
          for (int kk=k-1; kk<k+2; ++kk) {
            vol = vol + a_liquid_volume_fraction(ii, jj, kk);
          }

        bool flip = (vol >= 0.5*27.0);


        // --- ML interface-type classification -----------------------------
        // Class ids (Fortran convention, from get_class):
        //   0 no classification, 1 well-resolved, 2 ligament, 3 droplet,
        //   4 sheet/film, 5 ligament end, 6 sheet end
        int interface_class = 0;

        // The classifier needs a 5^3 stencil, so it is only defined two cells
        // in from the domain edge; elsewhere we fall back to PLICNET.
        const bool stencil_available =
            (i - 2 >= mesh.imino() && i + 2 <= mesh.imaxo() &&
             j - 2 >= mesh.jmino() && j + 2 <= mesh.jmaxo() &&
             k - 2 >= mesh.kmino() && k + 2 <= mesh.kmaxo());

        if (stencil_available) {
          ml_classifier::Stencil stencil;
          const IRL::Pt cell_center(mesh.xm(i), mesh.ym(j), mesh.zm(k));

          for (int ii = 0; ii < 5; ++ii) {
            for (int jj = 0; jj < 5; ++jj) {
              for (int kk = 0; kk < 5; ++kk) {
                const int gi = i + ii - 2;
                const int gj = j + jj - 2;
                const int gk = k + kk - 2;

                double vf = a_liquid_volume_fraction(gi, gj, gk);
                if (flip) vf = 1 - vf;
                stencil.f(ii, jj, kk) = vf;

                // Liquid centroid, made relative to the *centre* cell, scaled
                // by the cell size, then weighted by the volume fraction.
                IRL::Pt bary = a_liquid_centroid(gi, gj, gk);
                if (flip) bary = a_gas_centroid(gi, gj, gk);
                bary -= cell_center;
                bary[0] /= mesh.dx();
                bary[1] /= mesh.dy();
                bary[2] /= mesh.dz();
                bary *= vf;

                stencil.b(ii, jj, kk, 0) = bary[0];
                stencil.b(ii, jj, kk, 1) = bary[1];
                stencil.b(ii, jj, kk, 2) = bary[2];
              }
            }
          }
          interface_class = ml_classifier::get_class(stencil);
        }

        // Sheet/film gets the two-plane R2P treatment; everything else PLIC.
        const bool use_r2p = (interface_class == 4 || interface_class == 6);

        // Hybrid condition (from Fortran: norm_pos-norm_neg >= 0.5 OR ...)
        if(!use_r2p){//if ((n_pos - n_neg) >= 0.5 || (((n_pos - n_neg) < 0.5) && ((n_pos + n_neg) < 0.75))) {//
          // --- PLICNET Logic ---
          recon_method(i,j,k) = 0;
          num_planes(i, j, k) = 1;
          double moments[189] = {0};
          bool flip = false;
          double m000 = 0, m100 = 0, m010 = 0, m001 = 0;
          double center[3] = {0};
          int direction = 0, direction2 = 0;
          double n[3] = {0};
          IRL::Normal normal;
          double temp = 0;

          if (a_liquid_volume_fraction(i, j, k) >= 0.5) flip = true;
          
          if (flip) {
            for (int ii = i - 1; ii < i + 2; ++ii) {
              for (int jj = j - 1; jj < j + 2; ++jj) {
                for (int kk = k - 1; kk < k + 2; ++kk) {
                  int m_idx = 7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
                  moments[m_idx] = 1.0 - a_liquid_volume_fraction(ii, jj, kk);
                  moments[m_idx + 1] = (a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 2] = (a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 3] = (a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  moments[m_idx + 4] = (a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 5] = (a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 6] = (a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  
                  m000 = m000 + (moments[m_idx]);
                  m100 = m100 + (moments[m_idx + 1] + (ii - i)) * (moments[m_idx]);
                  m010 = m010 + (moments[m_idx + 2] + (jj - j)) * (moments[m_idx]);
                  m001 = m001 + (moments[m_idx + 3] + (kk - k)) * (moments[m_idx]);
                }
              }
            }
          } else {
            for (int ii = i - 1; ii < i + 2; ++ii) {
              for (int jj = j - 1; jj < j + 2; ++jj) {
                for (int kk = k - 1; kk < k + 2; ++kk) {
                  int m_idx = 7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
                  moments[m_idx] = a_liquid_volume_fraction(ii, jj, kk);
                  moments[m_idx + 1] = (a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 2] = (a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 3] = (a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  moments[m_idx + 4] = (a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 5] = (a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 6] = (a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  
                  m000 = m000 + (moments[m_idx]);
                  m100 = m100 + (moments[m_idx + 1] + (ii - i)) * (moments[m_idx]);
                  m010 = m010 + (moments[m_idx + 2] + (jj - j)) * (moments[m_idx]);
                  m001 = m001 + (moments[m_idx + 3] + (kk - k)) * (moments[m_idx]);
                }
              }
            }
          }
          
          center[0] = m100 / m000;
          center[1] = m010 / m000;
          center[2] = m001 / m000;
          
          plicnet::reflect_moments(moments, center, &direction, &direction2);
          plicnet::get_normal(moments, n);
          normal = IRL::Normal(n[0], n[1], n[2]);

          switch (direction2) {
            case 1: temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
            case 2: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp; break;
            case 3: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp; break;
            case 4: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp; temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
            case 5: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp; temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
          }

          switch (direction) {
            case 1: normal[0] = -normal[0]; break;
            case 2: normal[1] = -normal[1]; break;
            case 3: normal[2] = -normal[2]; break;
            case 4: normal[0] = -normal[0]; normal[1] = -normal[1]; break;
            case 5: normal[0] = -normal[0]; normal[2] = -normal[2]; break;
            case 6: normal[1] = -normal[1]; normal[2] = -normal[2]; break;
            case 7: normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; break;
          }

          if (!flip) {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
          }

          normal[0] = normal[0] * mesh.dx();
          normal[1] = normal[1] * mesh.dy();
          normal[2] = normal[2] * mesh.dz();
          normal.normalize();

          const IRL::Normal& n1 = normal;
          const double vf = a_liquid_volume_fraction(i, j, k);
          const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), 
              IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
          double distance = IRL::findDistanceOnePlane(cube, vf, n1);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal, distance));

        } 
        else 
        {

          double vol = 0;
          for (int ii=i-1; ii<i+2; ++ii)
            for (int jj=j-1; jj<j+2; ++jj)
            for (int kk=k-1; kk<k+2; ++kk) {
              vol = vol + a_liquid_volume_fraction(ii, jj, kk);
            }

          bool flip = (vol >= 0.5*27.0);

          recon_method(i, j, k) = 1;
          std::vector<IRL::Pt> points;
          // --- R2P3D Logic ---
          for (int ii = i - 1; ii < i + 2; ++ii) {
            for (int jj = j - 1; jj < j + 2; ++jj) {
              for (int kk = k - 1; kk < k + 2; ++kk) {
                const int ind = (ii - i + 1) * 9 + (jj - j + 1) * 3 + (kk - k + 1);
                stencil_cells[ind] = IRL::RectangularCuboid::fromBoundingPts(
                    IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                    IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
                double vol = stencil_cells[ind].calculateVolume();
                stencil_moments[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
                    IRL::VolumeMoments(a_liquid_volume_fraction(ii, jj, kk) * vol,
                                       a_liquid_centroid(ii, jj, kk)),
                    IRL::VolumeMoments(
                        (1.0 - a_liquid_volume_fraction(ii, jj, kk)) * vol,
                        a_gas_centroid(ii, jj, kk)));
                neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind),
                                       &stencil_cells[ind],
                                       &stencil_moments[ind]);
              }
            }
          }

          if (listed_moments(i, j, k).size() == 0) {
            // No advected interface, use MOF3D
            auto cell = IRL::RectangularCuboid::fromBoundingPts(
                IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
            double vol = cell.calculateVolume();
            IRL::SeparatedMoments<IRL::VolumeMoments> svm(
                IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                                   a_liquid_centroid(i, j, k)),
                IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                                   a_gas_centroid(i, j, k)));
            (*a_interface)(i, j, k) = IRL::reconstructionWithMOF3D(cell, svm);
            neighborhood.setSurfaceArea(getReconstructionSurfaceArea(cell, (*a_interface)(i, j, k)));
          } else {
            // Use AdvectedNormals
            (*a_interface)(i, j, k) = IRL::reconstructionWithAdvectedNormals(listed_moments(i, j, k), neighborhood);
            double area_sum = 0.0;
            for (const auto& moment : listed_moments(i, j, k)) {
              area_sum += moment.volumeMoments().volume();
            }
            neighborhood.setSurfaceArea(area_sum);
          }

          (*a_interface)(i, j, k) = reconstructionWithR2P3D(neighborhood, (*a_interface)(i, j, k));

          if ((*a_interface)(i, j, k).getNumberOfPlanes() == 1)
          {
            num_planes(i, j, k) = 1;
          }
          else if ((*a_interface)(i, j, k).getNumberOfPlanes() == 2)
          {
            num_planes(i, j, k) = 2;
          }
          else
          {
            num_planes(i, j, k) = 0;
          }
        }
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
}
































void R2PDistanceSolver(double VF_target, IRL::Pt bary_target, IRL::PlanarSeparator &a_interface, IRL::RectangularCuboid cell)
{
  IRL::Pt cell_centroid = cell.calculateCentroid();
  int sign = a_interface.isNotFlipped() ? 1 : -1;

  if (sign == -1)
  {
    bary_target = (cell_centroid - bary_target*VF_target);
    bary_target[0] = bary_target[0]/(1 - VF_target);
    bary_target[1] = bary_target[1]/(1 - VF_target);
    bary_target[2] = bary_target[2]/(1 - VF_target);
  }

  IRL::Normal n = a_interface[0].normal() - a_interface[1].normal();
  if (n.calculateMagnitude() < 1.0e-12)
  {
    n = a_interface[0].normal();
  }
  n.normalize();

  double t = IRL::dotProduct(bary_target, n) - IRL::dotProduct(cell_centroid, n);
  double dist1 = IRL::dotProduct(cell_centroid, a_interface[0].normal()) + t;
  double dist2 = IRL::dotProduct(cell_centroid, a_interface[1].normal()) - t;

  double side = (cell.calculateSideLength(0)+cell.calculateSideLength(1)+cell.calculateSideLength(2))/3.0;
  double tol = 1e-14;
  IRL::Pt bary;

  {
    int max_iter = 200;
    int iter = 0;
    double VF_cut = 0.0;
    double error = 1.0;

    auto setInterval = [&](double shift)
    {
      a_interface[0] = IRL::Plane(a_interface[0].normal(), dist1 + sign*shift);
      a_interface[1] = IRL::Plane(a_interface[1].normal(), dist2 + sign*shift);
      auto m = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, a_interface);
      bary = m.volume() > 1.0e-14*cell.calculateVolume() ? m.centroid() : cell_centroid;
      return m.volume() / cell.calculateVolume();
    };

    double VF_zero = setInterval(0.0);
    double interval_min = 0.0;
    double interval_max = sign*VF_zero > sign*VF_target ? -0.25*side : 0.25*side;
    double VF_bound = setInterval(interval_max);

    while (iter < max_iter && (VF_zero - VF_target)*(VF_bound - VF_target) > 0.0)
    {
      interval_min = interval_max;
      interval_max *= 2.0;
      VF_bound = setInterval(interval_max);
      ++iter;
      if (std::abs(interval_max) > 20.0*side) break;
    }
    if (interval_max < interval_min)
    {
      std::swap(interval_min, interval_max);
    }

    std::array<double, 3> bounding_values{{interval_min, 0.5*(interval_min + interval_max), interval_max}};

    VF_cut = setInterval(bounding_values[1]);
    error = std::abs(VF_cut - VF_target);

    iter = 0;
    while (error > tol && iter < max_iter)
    {
      if (sign*VF_cut < sign*VF_target)
      {
        bounding_values[0] = bounding_values[1];
      }
      else
      {
        bounding_values[2] = bounding_values[1];
      }
      bounding_values[1] = 0.5*(bounding_values[0] + bounding_values[2]);
      VF_cut = setInterval(bounding_values[1]);
      error = std::abs(VF_cut - VF_target);
      ++iter;
    }
    IRL::cleanReconstruction(cell, VF_target, &a_interface);

    if (iter >= max_iter)
    {
      std::cout << "R2PDistanceSolver: bisection stalled, VF = " << VF_cut << " target " << VF_target << std::endl;
    }
  }
}

// shape_out, when non-null, receives three scale-invariant descriptors built
// from the covariance eigenvalues this routine already computes (l0 >= l1 >= l2):
//   [0] linearity  = (l0-l1)/l0   cloud is a line   (ligament / edge-on)
//   [1] planarity  = (l1-l2)/l0   cloud is a sheet  (well-defined film)
//   [2] sphericity =  l2/l0       cloud is isotropic (orientation is noise)
// The eigenvector says where the surface points; these say how well determined
// that direction is. They must be computed here rather than separately, so the
// values match the training pipeline bit for bit -- data_gen.h's computePCA
// derives them from the same unnormalized covariance, and the ratios are
// unaffected by that normalization.
IRL::Normal PCA_Normal(const std::vector<IRL::Pt>& points, double* shape_out = nullptr) 
{
    if (shape_out) { shape_out[0] = 0.0; shape_out[1] = 0.0; shape_out[2] = 1.0; }

    using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using Vector3 = Eigen::Vector<double, 3>;
    using Matrix33 = Eigen::Matrix<double, 3, 3>;

    const size_t N = points.size();
    assert(N >= 6 && "At least 6 points required for constrained paraboloid fit.");

    // 1. Calculate Centroid
    Vector3 centroid = Vector3::Zero();
    for (const auto& pt : points) 
    {
        centroid += Vector3(pt[0], pt[1], pt[2]);
    }
    centroid = centroid / double(N);

    // 2. PCA to find the plane of best fit
    Matrix33 covariance = Matrix33::Zero();
    for (const auto& pt : points) 
    {
        Vector3 d = Vector3(pt[0], pt[1], pt[2]) - centroid;
        covariance += d * d.transpose();
    }
    
    Eigen::SelfAdjointEigenSolver<Matrix33> eigensolver(covariance);
    // The eigenvector with the smallest eigenvalue is the normal to the plane
    Vector3 local_z = eigensolver.eigenvectors().col(0).normalized();

    if (shape_out)
    {
        // Eigen returns eigenvalues in increasing order.
        const Vector3 ev = eigensolver.eigenvalues();
        const double l2 = std::max(0.0, ev(0));   // smallest
        const double l1 = std::max(0.0, ev(1));
        const double l0 = std::max(0.0, ev(2));   // largest
        if (l0 > 1.0e-30)
        {
            shape_out[0] = (l0 - l1) / l0;
            shape_out[1] = (l1 - l2) / l0;
            shape_out[2] = l2 / l0;
        }
    }
    
    // We want the normal to generally point towards the positive global hemisphere to maintain consistency
    //if (local_z.sum() < 0.0) local_z *= -1.0;
    IRL::Normal n = IRL::Normal(local_z[0],local_z[1],local_z[2]);
    n.normalize();

    return n;
}

void R2P3D_Net::getReconstruction(
const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::Pt>& a_liquid_centroid, const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
    const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
    const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface) {
  
  // Get mesh everything is living on.
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  
  // Container for moments from advection
  Data<IRL::ListedVolumeMoments<IRL::VolumeMomentsAndNormal>> listed_moments(&mesh);

  // 1. Back-Project Surface Elements
  for (int i = mesh.imino() + 1; i <= mesh.imaxo() - 1; ++i) {
    for (int j = mesh.jmino() + 1; j <= mesh.jmaxo() - 1; ++j) {
      for (int k = mesh.kmino() + 1; k <= mesh.kmaxo() - 1; ++k) {
        auto cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
            IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
        const auto localizer_link = IRL::LocalizerLinkFromLocalizedSeparatorLink(
            &a_localized_separator_link(i, j, k));
        for (IRL::UnsignedIndex_t n = 0; n < (*a_interface)(i, j, k).getNumberOfPlanes(); ++n) {
          IRL::Polygon interface_poly =
              IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(
                  cell, (*a_interface)(i, j, k), (*a_interface)(i, j, k)[n]);
          if (interface_poly.getNumberOfVertices() == 0) continue;
          
          for (IRL::UnsignedIndex_t tri = 0; tri < interface_poly.getNumberOfSimplicesInDecomposition(); ++tri) {
            IRL::Tri simplex = static_cast<IRL::Tri>(interface_poly.getSimplexFromDecomposition(tri));
            for (auto& vertex : simplex) {
              vertex = back_project_vertex(vertex, a_dt, a_U, a_V, a_W);
            }
            simplex.calculateAndSetPlaneOfExistence();
            auto new_moments =
                IRL::getVolumeMoments<IRL::TaggedAccumulatedListedVolumeMoments<
                    IRL::VolumeMomentsAndNormal>>(simplex, localizer_link);
            for (IRL::UnsignedIndex_t moment = 0; moment < new_moments.size(); ++moment) {
              auto index_for_tag = getIndexFromTag(mesh, new_moments.getTagForIndex(moment));
              listed_moments(index_for_tag[0], index_for_tag[1], index_for_tag[2]) +=
                  new_moments.getMomentsForIndex(moment);
            }
          }
        }
      }
    }
  }

  // 2. Clean Up Advected Elements
  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        const IRL::UnsignedIndex_t starting_length = listed_moments(i, j, k).size();
        for (IRL::UnsignedIndex_t n = starting_length - 1; n != static_cast<IRL::UnsignedIndex_t>(-1); --n) {
          IRL::VolumeMomentsAndNormal& moment = listed_moments(i, j, k)[n];
          moment.normalizeByVolume();
          moment.normal().normalize();
          if (moment.normal().calculateMagnitude() < 0.95) {
            listed_moments(i, j, k).erase(n);
          } else {
            moment.multiplyByVolume();
          }
        }
      }
    }
  }

  // 3. Zonghao's Colinearity Metric Allocation
  Data<double> norm_pos(&mesh);
  Data<double> norm_neg(&mesh);
  Data<double> tmp_pos(&mesh);
  Data<double> tmp_neg(&mesh);

  for (int i = mesh.imino(); i <= mesh.imaxo(); ++i) {
    for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j) {
      for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
        norm_pos(i, j, k) = 0.0;
        norm_neg(i, j, k) = 0.0;
        tmp_pos(i, j, k) = 0.0;
        tmp_neg(i, j, k) = 0.0;
      }
    }
  }

  // 4. Compute Colinearity Metric
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          continue;
        }
        
        double surface_area = 0.0;
        std::vector<IRL::Normal> normals_adj;
        std::vector<double> area_adj;

        for (int ii = i - 1; ii <= i + 1; ++ii) {
          for (int jj = j - 1; jj <= j + 1; ++jj) {
            for (int kk = k - 1; kk <= k + 1; ++kk) {
              for (IRL::UnsignedIndex_t ind = 0; ind < listed_moments(ii, jj, kk).size(); ++ind) {
                IRL::VolumeMomentsAndNormal moment = listed_moments(ii, jj, kk)[ind];
                double area = moment.volumeMoments().volume();
                IRL::Normal n_adj = moment.normal();
                n_adj.normalize();
                normals_adj.push_back(n_adj);
                area_adj.push_back(area);
                surface_area += area;
              }
            }
          }
        }

        if (surface_area > 0.0) {
          double surf_dot_pos_sum = 0.0;
          double surf_dot_neg_sum = 0.0;
          int size_adj = normals_adj.size();
          std::vector<double> norm_pos_loc(size_adj, 0.0);
          std::vector<double> norm_neg_loc(size_adj, 0.0);

          for (int n = 0; n < size_adj; ++n) {
            for (int nn = 0; nn < size_adj; ++nn) {
              if (n == nn) continue;
              double dot_result = normals_adj[n] * normals_adj[nn];
              if (dot_result >= 0.0) norm_pos_loc[n] += area_adj[nn] * dot_result;
              if (dot_result < 0.0)  norm_neg_loc[n] -= area_adj[nn] * dot_result;
            }
            norm_pos_loc[n] /= (surface_area - area_adj[n]);
            norm_neg_loc[n] /= (surface_area - area_adj[n]);
          }

          for (int n = 0; n < size_adj; ++n) {
            surf_dot_pos_sum += norm_pos_loc[n] * area_adj[n];
            surf_dot_neg_sum += norm_neg_loc[n] * area_adj[n];
          }
          norm_pos(i, j, k) = surf_dot_pos_sum / surface_area;
          norm_neg(i, j, k) = surf_dot_neg_sum / surface_area;
        }
      }
    }
  }

  // 5. Filter Metric
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          continue;
        }

        double surface_area = 0.0;
        for (int ii = i - 1; ii <= i + 1; ++ii) {
          for (int jj = j - 1; jj <= j + 1; ++jj) {
            for (int kk = k - 1; kk <= k + 1; ++kk) {
              double cell_area = 0.0;
              for (IRL::UnsignedIndex_t ind = 0; ind < listed_moments(ii, jj, kk).size(); ++ind) {
                cell_area += listed_moments(ii, jj, kk)[ind].volumeMoments().volume();
              }
              surface_area += cell_area;
              tmp_pos(i, j, k) += cell_area * norm_pos(ii, jj, kk);
              tmp_neg(i, j, k) += cell_area * norm_neg(ii, jj, kk);
            }
          }
        }
        if (surface_area > 0.0) {
          tmp_pos(i, j, k) /= surface_area;
          tmp_neg(i, j, k) /= surface_area;
        }
      }
    }
  }

  // Assign Filtered Metrics
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        norm_pos(i, j, k) = tmp_pos(i, j, k);
        norm_neg(i, j, k) = tmp_neg(i, j, k);
      }
    }
  }

  // 6. Hybrid R2PNET Main Evaluation and Reconstruction Loop
  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        if (a_liquid_volume_fraction(i, j, k) < IRL::global_constants::VF_LOW ||
            a_liquid_volume_fraction(i, j, k) > IRL::global_constants::VF_HIGH) {
          const double distance = std::copysign(1.0, a_liquid_volume_fraction(i, j, k) - 0.5);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
          continue;
        }
        branch(i,j,k) = 0;
        double n_pos = norm_pos(i, j, k);
        double n_neg = norm_neg(i, j, k);

        double vol = 0;
        for (int ii=i-1; ii<i+2; ++ii)
          for (int jj=j-1; jj<j+2; ++jj)
          for (int kk=k-1; kk<k+2; ++kk) {
            vol = vol + a_liquid_volume_fraction(ii, jj, kk);
          }

        bool flip = (vol >= 0.5*27.0);

        auto plicnet_normal = [&](void) -> IRL::Normal {
          double moments_p[189] = {0};
          double m000 = 0, m100 = 0, m010 = 0, m001 = 0;
          double center_p[3] = {0};
          int dir1 = 0, dir2 = 0;
          double n[3] = {0};
          double temp = 0;
          const bool flip_plic = (a_liquid_volume_fraction(i, j, k) >= 0.5);

          for (int ii = i - 1; ii < i + 2; ++ii)
          for (int jj = j - 1; jj < j + 2; ++jj)
          for (int kk = k - 1; kk < k + 2; ++kk) {
            const int idx = 7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
            if (flip_plic) {
              moments_p[idx  ] = 1.0 - a_liquid_volume_fraction(ii, jj, kk);
              moments_p[idx+1] = (a_gas_centroid(ii,jj,kk)[0]    - mesh.xm(ii)) / mesh.dx();
              moments_p[idx+2] = (a_gas_centroid(ii,jj,kk)[1]    - mesh.ym(jj)) / mesh.dy();
              moments_p[idx+3] = (a_gas_centroid(ii,jj,kk)[2]    - mesh.zm(kk)) / mesh.dz();
              moments_p[idx+4] = (a_liquid_centroid(ii,jj,kk)[0] - mesh.xm(ii)) / mesh.dx();
              moments_p[idx+5] = (a_liquid_centroid(ii,jj,kk)[1] - mesh.ym(jj)) / mesh.dy();
              moments_p[idx+6] = (a_liquid_centroid(ii,jj,kk)[2] - mesh.zm(kk)) / mesh.dz();
            } else {
              moments_p[idx  ] = a_liquid_volume_fraction(ii, jj, kk);
              moments_p[idx+1] = (a_liquid_centroid(ii,jj,kk)[0] - mesh.xm(ii)) / mesh.dx();
              moments_p[idx+2] = (a_liquid_centroid(ii,jj,kk)[1] - mesh.ym(jj)) / mesh.dy();
              moments_p[idx+3] = (a_liquid_centroid(ii,jj,kk)[2] - mesh.zm(kk)) / mesh.dz();
              moments_p[idx+4] = (a_gas_centroid(ii,jj,kk)[0]    - mesh.xm(ii)) / mesh.dx();
              moments_p[idx+5] = (a_gas_centroid(ii,jj,kk)[1]    - mesh.ym(jj)) / mesh.dy();
              moments_p[idx+6] = (a_gas_centroid(ii,jj,kk)[2]    - mesh.zm(kk)) / mesh.dz();
            }
            m000 += moments_p[idx];
            m100 += (moments_p[idx+1] + (ii - i)) * moments_p[idx];
            m010 += (moments_p[idx+2] + (jj - j)) * moments_p[idx];
            m001 += (moments_p[idx+3] + (kk - k)) * moments_p[idx];
          }

          center_p[0] = m100 / m000;
          center_p[1] = m010 / m000;
          center_p[2] = m001 / m000;

          plicnet::reflect_moments(moments_p, center_p, &dir1, &dir2);
          plicnet::get_normal(moments_p, n);
          IRL::Normal normal = IRL::Normal(n[0], n[1], n[2]);

          switch (dir2) {
            case 1: temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
            case 2: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp; break;
            case 3: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp; break;
            case 4: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp;
                    temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
            case 5: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp;
                    temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
          }
          switch (dir1) {
            case 1: normal[0]=-normal[0]; break;
            case 2: normal[1]=-normal[1]; break;
            case 3: normal[2]=-normal[2]; break;
            case 4: normal[0]=-normal[0]; normal[1]=-normal[1]; break;
            case 5: normal[0]=-normal[0]; normal[2]=-normal[2]; break;
            case 6: normal[1]=-normal[1]; normal[2]=-normal[2]; break;
            case 7: normal[0]=-normal[0]; normal[1]=-normal[1]; normal[2]=-normal[2]; break;
          }
          if (!flip_plic) { normal[0]=-normal[0]; normal[1]=-normal[1]; normal[2]=-normal[2]; }

          normal[0] *= mesh.dx();
          normal[1] *= mesh.dy();
          normal[2] *= mesh.dz();
          normal.normalize();
          return normal;
        };


        // --- ML interface-type classification -----------------------------
        // Class ids (Fortran convention, from get_class):
        //   0 no classification, 1 well-resolved, 2 ligament, 3 droplet,
        //   4 sheet/film, 5 ligament end, 6 sheet end
        int interface_class = 0;

        // The classifier needs a 5^3 stencil, so it is only defined two cells
        // in from the domain edge; elsewhere we fall back to PLICNET.
        const bool stencil_available =
            (i - 2 >= mesh.imino() && i + 2 <= mesh.imaxo() &&
             j - 2 >= mesh.jmino() && j + 2 <= mesh.jmaxo() &&
             k - 2 >= mesh.kmino() && k + 2 <= mesh.kmaxo());

        if (stencil_available) {
          ml_classifier::Stencil stencil;
          const IRL::Pt cell_center(mesh.xm(i), mesh.ym(j), mesh.zm(k));

          for (int ii = 0; ii < 5; ++ii) {
            for (int jj = 0; jj < 5; ++jj) {
              for (int kk = 0; kk < 5; ++kk) {
                const int gi = i + ii - 2;
                const int gj = j + jj - 2;
                const int gk = k + kk - 2;

                double vf = a_liquid_volume_fraction(gi, gj, gk);
                if (flip) vf = 1 - vf;
                stencil.f(ii, jj, kk) = vf;

                // Liquid centroid, made relative to the *centre* cell, scaled
                // by the cell size, then weighted by the volume fraction.
                IRL::Pt bary = a_liquid_centroid(gi, gj, gk);
                if (flip) bary = a_gas_centroid(gi, gj, gk);
                bary -= cell_center;
                bary[0] /= mesh.dx();
                bary[1] /= mesh.dy();
                bary[2] /= mesh.dz();
                bary *= vf;

                stencil.b(ii, jj, kk, 0) = bary[0];
                stencil.b(ii, jj, kk, 1) = bary[1];
                stencil.b(ii, jj, kk, 2) = bary[2];
              }
            }
          }
          interface_class = ml_classifier::get_class(stencil);
        }

        feature_class(i,j,k) = interface_class;
        // Sheet/film gets the two-plane R2P treatment; everything else PLIC.
        const bool use_r2p = (interface_class == 4 || interface_class == 6);

        // Hybrid condition (from Fortran: norm_pos-norm_neg >= 0.5 OR ...)
        if(!use_r2p){//if ((n_pos - n_neg) >= 0.5 || (((n_pos - n_neg) < 0.5) && ((n_pos + n_neg) < 0.75))) {//if(false){//
          // --- PLICNET Logic ---
          recon_method(i,j,k) = 0;
          num_planes(i, j, k) = 1;
          double moments[189] = {0};
          bool flip_plic = false;
          double m000 = 0, m100 = 0, m010 = 0, m001 = 0;
          double center[3] = {0};
          int direction = 0, direction2 = 0;
          double n[3] = {0};
          IRL::Normal normal;
          double temp = 0;

          if (a_liquid_volume_fraction(i, j, k) >= 0.5) flip_plic = true;
          
          if (flip_plic) {
            for (int ii = i - 1; ii < i + 2; ++ii) {
              for (int jj = j - 1; jj < j + 2; ++jj) {
                for (int kk = k - 1; kk < k + 2; ++kk) {
                  int m_idx = 7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
                  moments[m_idx] = 1.0 - a_liquid_volume_fraction(ii, jj, kk);
                  moments[m_idx + 1] = (a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 2] = (a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 3] = (a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  moments[m_idx + 4] = (a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 5] = (a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 6] = (a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  
                  m000 = m000 + (moments[m_idx]);
                  m100 = m100 + (moments[m_idx + 1] + (ii - i)) * (moments[m_idx]);
                  m010 = m010 + (moments[m_idx + 2] + (jj - j)) * (moments[m_idx]);
                  m001 = m001 + (moments[m_idx + 3] + (kk - k)) * (moments[m_idx]);
                }
              }
            }
          } else {
            for (int ii = i - 1; ii < i + 2; ++ii) {
              for (int jj = j - 1; jj < j + 2; ++jj) {
                for (int kk = k - 1; kk < k + 2; ++kk) {
                  int m_idx = 7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
                  moments[m_idx] = a_liquid_volume_fraction(ii, jj, kk);
                  moments[m_idx + 1] = (a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 2] = (a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 3] = (a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  moments[m_idx + 4] = (a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
                  moments[m_idx + 5] = (a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
                  moments[m_idx + 6] = (a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
                  
                  m000 = m000 + (moments[m_idx]);
                  m100 = m100 + (moments[m_idx + 1] + (ii - i)) * (moments[m_idx]);
                  m010 = m010 + (moments[m_idx + 2] + (jj - j)) * (moments[m_idx]);
                  m001 = m001 + (moments[m_idx + 3] + (kk - k)) * (moments[m_idx]);
                }
              }
            }
          }
          
          center[0] = m100 / m000;
          center[1] = m010 / m000;
          center[2] = m001 / m000;
          
          plicnet::reflect_moments(moments, center, &direction, &direction2);
          plicnet::get_normal(moments, n);
          normal = IRL::Normal(n[0], n[1], n[2]);
//std::cout << "i " << i << " j " << j << " k " << k << " " << normal << std::endl << std::endl;
          switch (direction2) {
            case 1: temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
            case 2: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp; break;
            case 3: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp; break;
            case 4: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp; temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
            case 5: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp; temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
          }

          switch (direction) {
            case 1: normal[0] = -normal[0]; break;
            case 2: normal[1] = -normal[1]; break;
            case 3: normal[2] = -normal[2]; break;
            case 4: normal[0] = -normal[0]; normal[1] = -normal[1]; break;
            case 5: normal[0] = -normal[0]; normal[2] = -normal[2]; break;
            case 6: normal[1] = -normal[1]; normal[2] = -normal[2]; break;
            case 7: normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; break;
          }

          if (!flip_plic) {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
          }

          normal[0] = normal[0] * mesh.dx();
          normal[1] = normal[1] * mesh.dy();
          normal[2] = normal[2] * mesh.dz();
          normal.normalize();

          const IRL::Normal& n1 = normal;
          const double vf = a_liquid_volume_fraction(i, j, k);
          const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), 
              IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
          double distance = IRL::findDistanceOnePlane(cube, vf, n1);
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal, distance));

        } 
        else 
        {
          IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
          neighborhood.resize(27);
          neighborhood.setCenterOfStencil(13);
          IRL::RectangularCuboid stencil_cells[27];
          IRL::SeparatedMoments<IRL::VolumeMoments> stencil_moments[27];

          recon_method(i, j, k) = 1;
          std::vector<IRL::Pt> points;
          // --- R2P3D Logic ---
          for (int ii = i - 1; ii < i + 2; ++ii) {
            for (int jj = j - 1; jj < j + 2; ++jj) {
              for (int kk = k - 1; kk < k + 2; ++kk) {
                const int ind = (ii - i + 1) * 9 + (jj - j + 1) * 3 + (kk - k + 1);
                stencil_cells[ind] = IRL::RectangularCuboid::fromBoundingPts(
                    IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk)),
                    IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1)));
                double vol = stencil_cells[ind].calculateVolume();
                stencil_moments[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
                    IRL::VolumeMoments(a_liquid_volume_fraction(ii, jj, kk) * vol,
                                        a_liquid_centroid(ii, jj, kk)),
                    IRL::VolumeMoments(
                        (1.0 - a_liquid_volume_fraction(ii, jj, kk)) * vol,
                        a_gas_centroid(ii, jj, kk)));
                neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind),
                                        &stencil_cells[ind],
                                        &stencil_moments[ind]);


                if (!flip)
                {
                  if (a_liquid_volume_fraction(ii, jj, kk) > IRL::global_constants::VF_LOW)
                  {
                    points.push_back(a_liquid_centroid(ii, jj, kk));
                  }
                }
                else
                {
                  if ((1-a_liquid_volume_fraction(ii, jj, kk)) > IRL::global_constants::VF_LOW)
                  {
                    points.push_back(a_gas_centroid(ii, jj, kk));
                  }
                }
              }
            }
          }

          //if (i == mesh.imin() || j == mesh.jmin() || k == mesh.kmin() || i == mesh.imax() || j == mesh.jmax() || k == mesh.kmax())
          {
            if (listed_moments(i, j, k).size() == 0) {
              // No advected interface, use MOF3D
              auto cell = IRL::RectangularCuboid::fromBoundingPts(
                  IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                  IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
              double vol = cell.calculateVolume();
              IRL::SeparatedMoments<IRL::VolumeMoments> svm(
                  IRL::VolumeMoments(a_liquid_volume_fraction(i, j, k) * vol,
                                    a_liquid_centroid(i, j, k)),
                  IRL::VolumeMoments((1.0 - a_liquid_volume_fraction(i, j, k)) * vol,
                                    a_gas_centroid(i, j, k)));
              (*a_interface)(i, j, k) = IRL::reconstructionWithMOF3D(cell, svm);
              neighborhood.setSurfaceArea(getReconstructionSurfaceArea(cell, (*a_interface)(i, j, k)));
            } else {
              // Use AdvectedNormals
              (*a_interface)(i, j, k) = IRL::reconstructionWithAdvectedNormals(listed_moments(i, j, k), neighborhood);
              double area_sum = 0.0;
              for (const auto& moment : listed_moments(i, j, k)) {
                area_sum += moment.volumeMoments().volume();
              }
              neighborhood.setSurfaceArea(area_sum);
            }

            (*a_interface)(i, j, k) = reconstructionWithR2P3D(neighborhood, (*a_interface)(i, j, k));
            //std::cout << (*a_interface)(i, j, k) << std::endl;
          }
          if (!(i == mesh.imin() || j == mesh.jmin() || k == mesh.kmin() || i == mesh.imax() || j == mesh.jmax() || k == mesh.kmax()))
          {
            double moments[189] = {0};
            double m000=0, m100=0, m010=0, m001=0;

            if (flip) {
              for (int ii=i-1; ii<i+2; ++ii)
              for (int jj=j-1; jj<j+2; ++jj)
              for (int kk=k-1; kk<k+2; ++kk) {
                const int idx = 7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k));
                moments[idx  ] = 1.0 - a_liquid_volume_fraction(ii,jj,kk);
                moments[idx+1] = (a_gas_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[idx+2] = (a_gas_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[idx+3] = (a_gas_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                moments[idx+4] = (a_liquid_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[idx+5] = (a_liquid_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[idx+6] = (a_liquid_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                m000 += moments[idx];
                m100 += (moments[idx+1]+(ii-i))*moments[idx];
                m010 += (moments[idx+2]+(jj-j))*moments[idx];
                m001 += (moments[idx+3]+(kk-k))*moments[idx];
              }
            } else {
              for (int ii=i-1; ii<i+2; ++ii)
              for (int jj=j-1; jj<j+2; ++jj)
              for (int kk=k-1; kk<k+2; ++kk) {
                const int idx = 7*((ii+1-i)*9+(jj+1-j)*3+(kk+1-k));
                moments[idx  ] = a_liquid_volume_fraction(ii,jj,kk);
                moments[idx+1] = (a_liquid_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[idx+2] = (a_liquid_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[idx+3] = (a_liquid_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                moments[idx+4] = (a_gas_centroid(ii,jj,kk)[0]-mesh.xm(ii))/mesh.dx();
                moments[idx+5] = (a_gas_centroid(ii,jj,kk)[1]-mesh.ym(jj))/mesh.dy();
                moments[idx+6] = (a_gas_centroid(ii,jj,kk)[2]-mesh.zm(kk))/mesh.dz();
                m000 += moments[idx];
                m100 += (moments[idx+1]+(ii-i))*moments[idx];
                m010 += (moments[idx+2]+(jj-j))*moments[idx];
                m001 += (moments[idx+3]+(kk-k))*moments[idx];
              }
            }

            double pca_shape[3] = {0.0, 0.0, 1.0};
            IRL::Normal dir = PCA_Normal(points, pca_shape);
            //if (flip) dir = -dir;
            IRL::Pt bary = IRL::Pt(m100/m000,m010/m000,m001/m000);
            double dot = IRL::dotProduct(dir,bary);
            if (dot < 0) dir = -dir;
            // const double eps = 1e-10;

            // if (dir[0] < -eps) {
            //     dir = -dir;
            // } else if (std::abs(dir[0]) <= eps) {
            //     if (dir[1] < -eps) {
            //         dir = -dir;
            //     } else if (std::abs(dir[1]) <= eps && dir[2] < -eps) {
            //         dir = -dir;
            //     }
            // }
            //std::cout << i << " " << j << " " << k << " dir " << dir << " bary " << bary << " dot " << dot << std::endl;
            double center[3];
            center[0] = dir[0];
            center[1] = dir[1];
            center[2] = dir[2];

            int direction = 0, direction2 = 0;
            r2pnet::reflect_moments(moments, center, &direction, &direction2);

            double temp;
            switch (direction) {
              case 1: center[0]=-center[0]; break;
              case 2: center[1]=-center[1]; break;
              case 3: center[2]=-center[2]; break;
              case 4: center[0]=-center[0]; center[1]=-center[1]; break;
              case 5: center[0]=-center[0]; center[2]=-center[2]; break;
              case 6: center[1]=-center[1]; center[2]=-center[2]; break;
              case 7: center[0]=-center[0]; center[1]=-center[1]; center[2]=-center[2]; break;
            }
            switch (direction2) {
              case 1: temp=center[0]; center[0]=center[1]; center[1]=temp; break;
              case 2: temp=center[1]; center[1]=center[2]; center[2]=temp; break;
              case 3: temp=center[0]; center[0]=center[2]; center[2]=temp; break;
              case 4: temp=center[0]; center[0]=center[1]; center[1]=temp;
                      temp=center[1]; center[1]=center[2]; center[2]=temp; break;
              case 5: temp=center[0]; center[0]=center[1]; center[1]=temp;
                      temp=center[0]; center[0]=center[2]; center[2]=temp; break;
            }

            // if (center[0] < 0)
            // {
            //   center[0]=-center[0]; center[1]=-center[1]; center[2]=-center[2];
            // }

            // 195 inputs: 189 moments + 3 PCA direction + 3 PCA shape
            // descriptors. The descriptors are scalar functions of the
            // covariance eigenvalues, so unlike the direction they are
            // invariant under the reflections and axis permutations
            // reflect_moments applies -- they are copied straight through with
            // no accompanying transform, exactly as the generator writes them.
            //double input[195] = {0};
            double input[192] = {0};
            std::copy(moments, moments + 189, input);
            input[189] = center[0];
            input[190] = center[1];
            input[191] = center[2];
            // input[192] = pca_shape[0];
            // input[193] = pca_shape[1];
            // input[194] = pca_shape[2];


            double n[6] = {0,0,0,0,0,0};
            r2pnet::get_normals(input, n);
            IRL::Normal normal1 = IRL::Normal(n[0], n[1], n[2]);
            IRL::Normal normal2 = IRL::Normal(n[3], n[4], n[5]);

            if (normal2.calculateMagnitude() < 0.5 && IRL::dotProduct(normal1,dir)>0)
            {
              //normal1=-normal1;
            }
            //std::cout << "i " << i << " j " << j << " k " << k << " " << normal1 << " " << normal2 << std::endl << std::endl;

            switch (direction2) {
              case 1: temp=normal1[0]; normal1[0]=normal1[1]; normal1[1]=temp; break;
              case 2: temp=normal1[1]; normal1[1]=normal1[2]; normal1[2]=temp; break;
              case 3: temp=normal1[0]; normal1[0]=normal1[2]; normal1[2]=temp; break;
              case 4: temp=normal1[1]; normal1[1]=normal1[2]; normal1[2]=temp;
                      temp=normal1[0]; normal1[0]=normal1[1]; normal1[1]=temp; break;
              case 5: temp=normal1[0]; normal1[0]=normal1[2]; normal1[2]=temp;
                      temp=normal1[0]; normal1[0]=normal1[1]; normal1[1]=temp; break;
            }
            switch (direction) {
              case 1: normal1[0]=-normal1[0]; break;
              case 2: normal1[1]=-normal1[1]; break;
              case 3: normal1[2]=-normal1[2]; break;
              case 4: normal1[0]=-normal1[0]; normal1[1]=-normal1[1]; break;
              case 5: normal1[0]=-normal1[0]; normal1[2]=-normal1[2]; break;
              case 6: normal1[1]=-normal1[1]; normal1[2]=-normal1[2]; break;
              case 7: normal1[0]=-normal1[0]; normal1[1]=-normal1[1]; normal1[2]=-normal1[2]; break;
            }

            switch (direction2) {
              case 1: temp=normal2[0]; normal2[0]=normal2[1]; normal2[1]=temp; break;
              case 2: temp=normal2[1]; normal2[1]=normal2[2]; normal2[2]=temp; break;
              case 3: temp=normal2[0]; normal2[0]=normal2[2]; normal2[2]=temp; break;
              case 4: temp=normal2[1]; normal2[1]=normal2[2]; normal2[2]=temp;
                      temp=normal2[0]; normal2[0]=normal2[1]; normal2[1]=temp; break;
              case 5: temp=normal2[0]; normal2[0]=normal2[2]; normal2[2]=temp;
                      temp=normal2[0]; normal2[0]=normal2[1]; normal2[1]=temp; break;
            }
            switch (direction) {
              case 1: normal2[0]=-normal2[0]; break;
              case 2: normal2[1]=-normal2[1]; break;
              case 3: normal2[2]=-normal2[2]; break;
              case 4: normal2[0]=-normal2[0]; normal2[1]=-normal2[1]; break;
              case 5: normal2[0]=-normal2[0]; normal2[2]=-normal2[2]; break;
              case 6: normal2[1]=-normal2[1]; normal2[2]=-normal2[2]; break;
              case 7: normal2[0]=-normal2[0]; normal2[1]=-normal2[1]; normal2[2]=-normal2[2]; break;
            }

            bool one_plane = false;

            //std::cout << "mag " << normal2.calculateMagnitude() << std::endl;
            // normal1[0]=-0.577350269189626;
            // normal1[1]=-0.577350269189626;
            // normal1[2]=-0.577350269189626;
            //normal2=-normal1;
            if (normal2.calculateMagnitude() < 0.85 || normal1.calculateMagnitude() < 0.85)
            {
              one_plane = true;
            }
            // bool te = false;
            // if ((*a_interface)(i, j, k).getNumberOfPlanes() == 1)
            // {
            //   te = true;
            //   std::cout << "standard " << (*a_interface)(i, j, k) << std::endl;
            //   std::cout << normal1 << " " << normal2 << std::endl;
            // }


            if (!one_plane)
            {
              (*a_interface)(i, j, k).setNumberOfPlanes(2);
              branch(i,j,k) = 2;
              normal1[0] *= mesh.dx();
              normal1[1] *= mesh.dy();
              normal1[2] *= mesh.dz();
              normal1.normalize();
              normal2[0] *= mesh.dx();
              normal2[1] *= mesh.dy();
              normal2[2] *= mesh.dz();
              normal2.normalize();

              int flip_i = 1;
              if (flip) flip_i = -1;
              if (!flip) normal1=-normal1;
              if (!flip) normal2=-normal2;
              IRL::Pt bary = a_liquid_centroid(i, j, k);
              IRL::Pt bary1 = a_gas_centroid(i, j, k);

              const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
              (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(normal1,0),IRL::Plane(normal2,0),flip_i);
              R2PDistanceSolver(a_liquid_volume_fraction(i, j, k),bary,(*a_interface)(i, j, k),cube);
              //R2PDistanceSolver2(neighborhood,(*a_interface)(i, j, k));
            }
            // if (!one_plane)
            // {
            //   (*a_interface)(i, j, k).setNumberOfPlanes(2);
            //   branch(i,j,k) = 2;
            //   normal1[0] *= mesh.dx();
            //   normal1[1] *= mesh.dy();
            //   normal1[2] *= mesh.dz();
            //   normal1.normalize();
            //   normal2[0] *= mesh.dx();
            //   normal2[1] *= mesh.dy();
            //   normal2[2] *= mesh.dz();
            //   normal2.normalize();

            //   int flip_i = 1;
            //   if (flip) flip_i = -1;
            //   if (!flip) normal1=-normal1;
            //   if (!flip) normal2=-normal2;
            //   // normal1=IRL::Normal(1,1,1);
            //   // normal2=IRL::Normal(-1,-1,-1);
            //   // normal1.normalize();
            //   // normal2.normalize();
            //   // if (IRL::dotProduct(normal1,(*a_interface)(i, j, k)[0].normal()) < 0)
            //   // {
            //   //   normal1=-normal1;
            //   //   normal2=-normal2;
            //   // }
            //   IRL::Pt bary = a_liquid_centroid(i, j, k);
            //   IRL::Pt bary1 = a_gas_centroid(i, j, k);

            //   const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
            //   (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(normal1,0),IRL::Plane(normal2,0),flip_i);
            //   //std::cout << "n0 " << normal1 << " n1 " << normal2 << " flip " << flip_i << " VF_target " << a_liquid_volume_fraction(i, j, k) << " bary_target " << bary << " cell vertices " << cube << std::endl;
            //   //R2PDistanceSolver(a_liquid_volume_fraction(i, j, k),bary,(*a_interface)(i, j, k),cube);
            //   //R2PDistanceSolver2(a_liquid_volume_fraction(i, j, k),bary,bary1,(*a_interface)(i, j, k),cube);
            //   R2PDistanceSolver2(neighborhood,(*a_interface)(i, j, k));
            //   //std::cout << "returned planes " << (*a_interface)(i, j, k) << std::endl;
            //   // if ((*a_interface)(i, j, k)[1].normal().calculateMagnitude() < IRL::global_constants::VF_LOW)
            //   // {
            //   //   one_plane = true;
            //   // }
            // }
            // if (!one_plane)
            // {
            //   (*a_interface)(i, j, k).setNumberOfPlanes(2);
              
            //   // Scale and normalize NN normals based on mesh
            //   normal1[0] *= mesh.dx();
            //   normal1[1] *= mesh.dy();
            //   normal1[2] *= mesh.dz();
            //   normal1.normalize();
              
            //   normal2[0] *= mesh.dx();
            //   normal2[1] *= mesh.dy();
            //   normal2[2] *= mesh.dz();
            //   normal2.normalize();

            //   int flip_i = 1;
            //   if (flip) flip_i = -1;
            //   if (!flip) normal1=-normal1;
            //   if (!flip) normal2=-normal2;

            //   const double target_vf = a_liquid_volume_fraction(i, j, k);
            //   const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(
            //       IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), 
            //       IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));

            //   // Because the NN doesn't inherently partition centroids, 
            //   // we utilize the standard liquid/gas centroids of the center cell.
            //   IRL::Pt pt1 = a_liquid_centroid(i, j, k);
            //   IRL::Pt pt2 = a_gas_centroid(i, j, k);

            //   IRL::PlanarSeparator best_separator;
            //   double min_err = DBL_MAX;

            //   // Helper lambda to mimic `constructSeparatorAttempt` and `checkIfBest`
            //   auto test_permutation = [&](const IRL::Normal& n0, const IRL::Pt& p0,
            //                               const IRL::Normal& n1, const IRL::Pt& p1,
            //                               double flip_cut) {
            //       // 1. Construct Separator Attempt
            //       IRL::PlanarSeparator attempt = IRL::PlanarSeparator::fromTwoPlanes(
            //           IRL::Plane(n0, n0 * p0),
            //           IRL::Plane(n1, n1 * p1), 
            //           flip_cut);

            //       // 2. Find volume conserving distance
            //       IRL::Pt bary = a_liquid_centroid(i, j, k);
            //       if (flip_cut==-1) bary = a_gas_centroid(i, j, k);
            //       R2PDistanceSolver(target_vf,bary,attempt,cube);

            //       // 3. Accumulate Error in Centroids across the Neighborhood
            //       double err = 0.0;
            //       for (const auto& cell_grouped_moments : neighborhood) {
            //           auto svm = cell_grouped_moments.calculateNormalizedVolumeMoments(attempt);
            //           double cell_volume = cell_grouped_moments.getStoredMoments()[0].volume() +
            //                                cell_grouped_moments.getStoredMoments()[1].volume();
            //           double cell_VF = svm[0].volume() / cell_volume;

            //           // Fix bounding volume fractions
            //           if (cell_VF < IRL::global_constants::VF_LOW) {
            //               svm[0].centroid() = neighborhood.getCenterCellStoredMoments()[0].centroid();
            //           }
            //           if (cell_VF > IRL::global_constants::VF_HIGH) {
            //               svm[1].centroid() = neighborhood.getCenterCellStoredMoments()[1].centroid();
            //           }

            //           // Accumulate Liquid & Gas errors
            //           if (cell_grouped_moments.getStoredMoments()[0].volume() / cell_volume > IRL::global_constants::VF_LOW) {
            //               err += IRL::magnitude(cell_grouped_moments.getStoredMoments()[0].centroid() - svm[0].centroid());
            //           }
            //           if (cell_grouped_moments.getStoredMoments()[1].volume() / cell_volume > IRL::global_constants::VF_LOW) {
            //               err += IRL::magnitude(cell_grouped_moments.getStoredMoments()[1].centroid() - svm[1].centroid());
            //           }
            //       }

            //       // 4. Update the best configuration if this attempt has lower error
            //       if (err < min_err) {
            //           min_err = err;
            //           best_separator = attempt;
            //       }
            //   };

            //   // Test all four permutations of normals, centroids, and flip direction
            //   test_permutation(normal1, pt1, normal2, pt2, 1.0);
            //   test_permutation(normal1, pt1, normal2, pt2, -1.0);
            //   test_permutation(normal1, pt2, normal2, pt1, 1.0);
            //   test_permutation(normal1, pt2, normal2, pt1, -1.0);
            //   (*a_interface)(i, j, k) = best_separator;
            //   // IRL::Pt bary = a_liquid_centroid(i, j, k);
            //   // if (flip) bary = a_gas_centroid(i, j, k);
            //   // const IRL::RectangularCuboid& cube1 = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
            //   // (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(normal1,0),IRL::Plane(normal2,0),flip_i);

            //   // R2PDistanceSolver(a_liquid_volume_fraction(i, j, k),bary,(*a_interface)(i, j, k),cube);
            // }
            // if (one_plane)
            // {
            //   branch(i,j,k) = 1;
            //   (*a_interface)(i, j, k).setNumberOfPlanes(1);
            //   if (normal2.calculateMagnitude() < normal1.calculateMagnitude())
            //   {
            //     normal1[0] *= mesh.dx();
            //     normal1[1] *= mesh.dy();
            //     normal1[2] *= mesh.dz();
            //     normal1.normalize();
            //     const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
            //     if (!flip) normal1=-normal1;
            //     if (IRL::dotProduct(normal1,(a_liquid_centroid(i,j,k)-cube.calculateCentroid())) > 0)
            //     {
            //       normal1=-normal1;
            //     }
            //     double distance = IRL::findDistanceOnePlane(cube, a_liquid_volume_fraction(i, j, k), normal1);
            //     (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal1,distance));
            //   }
            //   else
            //   {
            //     normal2[0] *= mesh.dx();
            //     normal2[1] *= mesh.dy();
            //     normal2[2] *= mesh.dz();
            //     normal2.normalize();
            //     const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)), IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
            //     if (!flip) normal2=-normal2;
            //     if (IRL::dotProduct(normal2,(a_liquid_centroid(i,j,k)-cube.calculateCentroid())) > 0)
            //     {
            //       normal2=-normal2;
            //     }
            //     double distance = IRL::findDistanceOnePlane(cube, a_liquid_volume_fraction(i, j, k), normal2);
            //     (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal2,distance));
            //   }
            // }
            if (one_plane)
            {
              const IRL::RectangularCuboid& cube = IRL::RectangularCuboid::fromBoundingPts(
                  IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                  IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
              const double target_vf   = a_liquid_volume_fraction(i, j, k);
              const IRL::Pt  target_liq = a_liquid_centroid(i, j, k);
              const IRL::Pt  target_gas = a_gas_centroid(i, j, k);
              const IRL::Pt  cell_ctr   = cube.calculateCentroid();

              // Build a volume-conserving separator from a raw normal and score it by
              // how well it reproduces the cell's liquid/gas centroids.
              auto build_and_score = [&](IRL::Normal nrm, IRL::PlanarSeparator* out) -> double {
                if (nrm.calculateMagnitude() < 0.5) return DBL_MAX;
                nrm.normalize();
                const double d = IRL::findDistanceOnePlane(cube, target_vf, nrm);
                *out = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(nrm, d));

                auto svm = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(cube, *out);
                const double cell_vol = cube.calculateVolume();
                const double vf_out   = svm[0].volume() / cell_vol;
                // Reject anything that failed to hit the target volume fraction.
                if (std::abs(vf_out - target_vf) > 1.0e-6) return DBL_MAX;

                double err = 0.0;
                if (target_vf > IRL::global_constants::VF_LOW)
                  err += IRL::magnitude(target_liq - svm[0].centroid());
                if (target_vf < IRL::global_constants::VF_HIGH)
                  err += IRL::magnitude(target_gas - svm[1].centroid());
                return err;
              };

              // --- Candidate 1: the surviving NN normal (existing behaviour) ---
              IRL::Normal nn_normal = (normal2.calculateMagnitude() < normal1.calculateMagnitude())
                                          ? normal1 : normal2;
              nn_normal[0] *= mesh.dx();
              nn_normal[1] *= mesh.dy();
              nn_normal[2] *= mesh.dz();
              if (nn_normal.calculateMagnitude() > 0.0) nn_normal.normalize();
              if (!flip) nn_normal = -nn_normal;
              if (IRL::dotProduct(nn_normal, (target_liq - cell_ctr)) > 0) nn_normal = -nn_normal;

              IRL::PlanarSeparator sep_nn;
              const double err_nn = build_and_score(nn_normal, &sep_nn);

              // --- Candidate 2: PLICNet ---
              IRL::PlanarSeparator sep_plic;
              const double err_plic = build_and_score(plicnet_normal(), &sep_plic);

              // --- Pick the better one ---
              if (err_plic < err_nn) {             
                branch(i,j,k) = 1;
                (*a_interface)(i, j, k) = sep_plic;
                recon_method(i, j, k) = 0;   // fell back to PLICNet
              } else {branch(i,j,k) = 3;
                (*a_interface)(i, j, k) = sep_nn;
                recon_method(i, j, k) = 1;
              }
            }
            //if(te) std::cout << "ML " << (*a_interface)(i, j, k) << std::endl << std::endl << std::endl << std::endl;
          }
          //std::cout << (*a_interface)(i, j, k) << std::endl << std::endl << std::endl << std::endl;
          // (*a_interface)(i, j, k) = reconstructionWithR2P3D(neighborhood, (*a_interface)(i, j, k));
          // std::cout << (*a_interface)(i, j, k) << std::endl << std::endl << std::endl;
          
          if ((*a_interface)(i, j, k).getNumberOfPlanes() == 1)
          {
            num_planes(i, j, k) = 1;
          }
          else if ((*a_interface)(i, j, k).getNumberOfPlanes() == 2)
          {
            num_planes(i, j, k) = 2;
          }
          else
          {
            num_planes(i, j, k) = 0;
          }
        }
      }
    }
  }
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);

  // --- pass 2: paraboloid refinement ---
  r2ppass::Options parab_opt;
  const r2ppass::Stats parab_stats =
      r2ppass::run(a_liquid_volume_fraction, a_liquid_centroid, a_interface,
                   &branch, parab_opt);

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