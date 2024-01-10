// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2019 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_INTERFACE_RECONSTRUCTION_METHODS_RECONSTRUCTION_INTERFACE_TPP_
#define IRL_INTERFACE_RECONSTRUCTION_METHODS_RECONSTRUCTION_INTERFACE_TPP_

namespace IRL {

template <class CellType>
PlanarSeparator reconstructionWithR2P2D(
    const R2PNeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  cleanReconstruction(
      a_neighborhood_geometry.getCenterCell(),
      (a_neighborhood_geometry.getCenterCellStoredMoments())[0].volume() /
          a_neighborhood_geometry.getCenterCell().calculateVolume(),
      &a_initial_reconstruction);
  if (a_initial_reconstruction.getNumberOfPlanes() == 1) {
    R2P_2D1P<CellType> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  } else {
    R2P_2D2P<CellType> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  }
}

template <class CellType>
PlanarSeparator reconstructionWithR2P3D(
    const R2PNeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  cleanReconstruction(
      a_neighborhood_geometry.getCenterCell(),
      (a_neighborhood_geometry.getCenterCellStoredMoments())[0].volume() /
          a_neighborhood_geometry.getCenterCell().calculateVolume(),
      &a_initial_reconstruction);
  if (a_initial_reconstruction.getNumberOfPlanes() == 1) {
    R2P_3D1P<CellType> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  } else {
    R2P_3D2P<CellType> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  }
}

template <class CellType>
PlanarSeparator reconstructionWithR2P3D(
    const R2PNeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction,
    const R2PWeighting& a_r2p_weighting) {
  cleanReconstruction(
      a_neighborhood_geometry.getCenterCell(),
      (a_neighborhood_geometry.getCenterCellStoredMoments())[0].volume() /
          a_neighborhood_geometry.getCenterCell().calculateVolume(),
      &a_initial_reconstruction);
  if (a_initial_reconstruction.getNumberOfPlanes() == 1) {
    R2P_3D1P<CellType> r2p_system;
    r2p_system.setCostFunctionBehavior(a_r2p_weighting);
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  } else {
    R2P_3D2P<CellType> r2p_system;
    r2p_system.setCostFunctionBehavior(a_r2p_weighting);
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  }
}

template <class CellType>
PlanarSeparator reconstructionWithR2P3D(
    const R2PNeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction,
    const OptimizationBehavior& a_optimization_behavior,    
    const R2PWeighting& a_r2p_weighting) {
  cleanReconstruction(
      a_neighborhood_geometry.getCenterCell(),
      (a_neighborhood_geometry.getCenterCellStoredMoments())[0].volume() /
          a_neighborhood_geometry.getCenterCell().calculateVolume(),
      &a_initial_reconstruction);
  if (a_initial_reconstruction.getNumberOfPlanes() == 1) {
    R2P_3D1P<CellType> r2p_system;
    r2p_system.setOptimizationBehavior(a_optimization_behavior);
    r2p_system.setCostFunctionBehavior(a_r2p_weighting);
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  } else {
    R2P_3D2P<CellType> r2p_system;
    r2p_system.setOptimizationBehavior(a_optimization_behavior);
    r2p_system.setCostFunctionBehavior(a_r2p_weighting);
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  }
}

PlanarSeparator reconstructionWithELVIRA2D(
    const ELVIRANeighborhood& a_neighborhood_geometry) {
  ELVIRA_2D elvira_system;
  return elvira_system.solve(&a_neighborhood_geometry);
}

PlanarSeparator reconstructionWithELVIRA3D(
    const ELVIRANeighborhood& a_neighborhood_geometry) {
  ELVIRA_3D elvira_system;
  return elvira_system.solve(&a_neighborhood_geometry);
}

template <class CellType>
PlanarSeparator reconstructionWithML(/*const ELVIRANeighborhood& a_neighborhood_geometry, */const LVIRANeighborhood<CellType>& a_neighborhood_geometry, const R2PNeighborhood<CellType>& r2pnh, const double* a_liquid_centroids, const double* a_gas_centroids, PlanarSeparator p, int* flag) 
{
  auto n = IRL::Normal();
  auto n2 = IRL::Normal();
  std::vector<double> fractions;
  auto sm = IRL::spatial_moments();

  bool flip = false;
  if (a_neighborhood_geometry.getCenterCellStoredMoments() > 0.5)
  {
    flip = true;
  }
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        double dx = a_neighborhood_geometry.getCell(k*9+j*3+i).calculateSideLength(0);
        double dy = a_neighborhood_geometry.getCell(k*9+j*3+i).calculateSideLength(1);
        double dz = a_neighborhood_geometry.getCell(k*9+j*3+i).calculateSideLength(2);
        if (!flip)
        {
          fractions.push_back(a_neighborhood_geometry.getStoredMoments(k*9+j*3+i));
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
          //fractions.push_back((a_gas_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          //fractions.push_back((a_gas_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          //fractions.push_back((a_gas_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
        }
        else
        {
          fractions.push_back(1 - a_neighborhood_geometry.getStoredMoments(k*9+j*3+i));
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
          //fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          //fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          //fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
        }
      }
    }
  }

  std::vector<double> center = sm.get_mass_centers(fractions);
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
              fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
              fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
              fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
              fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
              fractions[7*(2*9+j*3+k)+4] = -temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
              fractions[7*(2*9+j*3+k)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
              fractions[7*(2*9+j*3+k)+6] = temp;*/
          }
          else if (i == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
              //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
              fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
              fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
              fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
              fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
              fractions[7*(i*9+2*3+k)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
              fractions[7*(i*9+2*3+k)+5] = -temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
              fractions[7*(i*9+2*3+k)+6] = temp;*/
          }
          else if (j == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
              //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
              fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
              fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
              fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
              fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
              fractions[7*(i*9+j*3+2)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
              fractions[7*(i*9+j*3+2)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
              fractions[7*(i*9+j*3+2)+6] = -temp;*/
          }
          else if (k == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
              //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
              fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
              fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
              fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
              fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
              fractions[7*(2*9+j*3+k)+4] = -temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
              fractions[7*(2*9+j*3+k)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
              fractions[7*(2*9+j*3+k)+6] = temp;*/
          }
          else if (i == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
              //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
              fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
              fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
              fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
              fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
              fractions[7*(i*9+2*3+k)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
              fractions[7*(i*9+2*3+k)+5] = -temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
              fractions[7*(i*9+2*3+k)+6] = temp;*/
          }
          else if (j == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
              //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
              fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
              fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
              fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
              fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
              fractions[7*(2*9+j*3+k)+4] = -temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
              fractions[7*(2*9+j*3+k)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
              fractions[7*(2*9+j*3+k)+6] = temp;*/
          }
          else if (i == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
              //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
              fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
              fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
              fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
              fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
              fractions[7*(i*9+j*3+2)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
              fractions[7*(i*9+j*3+2)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
              fractions[7*(i*9+j*3+2)+6] = -temp;*/
          }
          else if (k == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
              //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
              fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
              fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
              fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
              fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
              fractions[7*(i*9+2*3+k)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
              fractions[7*(i*9+2*3+k)+5] = -temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
              fractions[7*(i*9+2*3+k)+6] = temp;*/
          }
          else if (j == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
              //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
              fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
              fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
              fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
              fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
              fractions[7*(i*9+j*3+2)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
              fractions[7*(i*9+j*3+2)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
              fractions[7*(i*9+j*3+2)+6] = -temp;*/
          }
          else if (k == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
              //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(2*9+j*3+k)+0];
              fractions[/*7*/4*(2*9+j*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(2*9+j*3+k)+1];
              fractions[/*7*/4*(2*9+j*3+k)+1] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(2*9+j*3+k)+2];
              fractions[/*7*/4*(2*9+j*3+k)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(2*9+j*3+k)+3];
              fractions[/*7*/4*(2*9+j*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = -fractions[7*(2*9+j*3+k)+4];
              fractions[7*(2*9+j*3+k)+4] = -temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(2*9+j*3+k)+5];
              fractions[7*(2*9+j*3+k)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(2*9+j*3+k)+6];
              fractions[7*(2*9+j*3+k)+6] = temp;*/
          }
          else if (i == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+1] = -fractions[/*7*/4*(i*9+j*3+k)+1];
              //fractions[7*(i*9+j*3+k)+4] = -fractions[7*(i*9+j*3+k)+4];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+2*3+k)+0];
              fractions[/*7*/4*(i*9+2*3+k)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+2*3+k)+1];
              fractions[/*7*/4*(i*9+2*3+k)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+2*3+k)+2];
              fractions[/*7*/4*(i*9+2*3+k)+2] = -temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = fractions[/*7*/4*(i*9+2*3+k)+3];
              fractions[/*7*/4*(i*9+2*3+k)+3] = temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+2*3+k)+4];
              fractions[7*(i*9+2*3+k)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+2*3+k)+5];
              fractions[7*(i*9+2*3+k)+5] = -temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = fractions[7*(i*9+2*3+k)+6];
              fractions[7*(i*9+2*3+k)+6] = temp;*/
          }
          else if (j == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+2] = -fractions[/*7*/4*(i*9+j*3+k)+2];
              //fractions[7*(i*9+j*3+k)+5] = -fractions[7*(i*9+j*3+k)+5];
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
              double temp = fractions[/*7*/4*(i*9+j*3+k)+0];
              fractions[/*7*/4*(i*9+j*3+k)+0] = fractions[/*7*/4*(i*9+j*3+2)+0];
              fractions[/*7*/4*(i*9+j*3+2)+0] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+1];
              fractions[/*7*/4*(i*9+j*3+k)+1] = fractions[/*7*/4*(i*9+j*3+2)+1];
              fractions[/*7*/4*(i*9+j*3+2)+1] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+2];
              fractions[/*7*/4*(i*9+j*3+k)+2] = fractions[/*7*/4*(i*9+j*3+2)+2];
              fractions[/*7*/4*(i*9+j*3+2)+2] = temp;
              temp = fractions[/*7*/4*(i*9+j*3+k)+3];
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+2)+3];
              fractions[/*7*/4*(i*9+j*3+2)+3] = -temp;
              /*temp = fractions[7*(i*9+j*3+k)+4];
              fractions[7*(i*9+j*3+k)+4] = fractions[7*(i*9+j*3+2)+4];
              fractions[7*(i*9+j*3+2)+4] = temp;
              temp = fractions[7*(i*9+j*3+k)+5];
              fractions[7*(i*9+j*3+k)+5] = fractions[7*(i*9+j*3+2)+5];
              fractions[7*(i*9+j*3+2)+5] = temp;
              temp = fractions[7*(i*9+j*3+k)+6];
              fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+2)+6];
              fractions[7*(i*9+j*3+2)+6] = -temp;*/
          }
          else if (k == 1)
          {
              fractions[/*7*/4*(i*9+j*3+k)+3] = -fractions[/*7*/4*(i*9+j*3+k)+3];
              //fractions[7*(i*9+j*3+k)+6] = -fractions[7*(i*9+j*3+k)+6];
          }
          }
      }
      }
  }
   
  //double inter = b.get_normal_loss(fractions);
  //if ((flag[0] != 2) || (inter > 0.015 && flag[0] == 2))
  {
    flag[0] = 1;
    //return reconstructionWithLVIRA3D(a_neighborhood_geometry, p);
    //return reconstructionWithR2P3D(r2pnh, p);
  }

  flag[0] = 0;
  n = t.get_normal(fractions);
  n.normalize();
  /*if (n[1] > 0.9995)
  {
    n[0] = 0;
    n[1] = 1;
    n[2] = 0;
  }
  else if (-n[1] > 0.9995)
  {
    n[0] = 0;
    n[1] = -1;
    n[2] = 0;
  }*/
  switch (direction)
  {
    case 1:
      n[0] = -n[0];
      break;
    case 2:
      n[1] = -n[1];
      break;
    case 3:
      n[2] = -n[2];
      break;
    case 4:
      n[0] = -n[0];
      n[1] = -n[1];
      break;
    case 5:
      n[0] = -n[0];
      n[2] = -n[2];
      break;
    case 6:
      n[1] = -n[1];
      n[2] = -n[2];
      break;
    case 7:
      n[0] = -n[0];
      n[1] = -n[1];
      n[2] = -n[2];
      break;
  }

  if (!flip)
  {
    n[0] = -n[0];
    n[1] = -n[1];
    n[2] = -n[2];
  }

  previous = n;
  const IRL::Normal& n1 = n;
  const double d = a_neighborhood_geometry.getCenterCellStoredMoments();
  const IRL::RectangularCuboid& cube = a_neighborhood_geometry.getCenterCell();
  double distance = IRL::findDistanceOnePlane(cube, d, n1);
  return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, distance));
}

template <class CellType>
PlanarSeparator reconstructionWithML2(/*const ELVIRANeighborhood& a_neighborhood_geometry, */const LVIRANeighborhood<CellType>& a_neighborhood_geometry, const double* a_liquid_centroids, const double* a_gas_centroids, PlanarSeparator p, int* flag) 
{
  auto n = IRL::Normal();
  auto n2 = IRL::Normal();
  std::vector<double> fractions;
  auto sm = IRL::spatial_moments();

  bool flip = false;
  if (a_neighborhood_geometry.getCenterCellStoredMoments() > 0.5)
  {
    flip = true;
  }
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        double dx = a_neighborhood_geometry.getCell(k*9+j*3+i).calculateSideLength(0);
        double dy = a_neighborhood_geometry.getCell(k*9+j*3+i).calculateSideLength(1);
        double dz = a_neighborhood_geometry.getCell(k*9+j*3+i).calculateSideLength(2);
        if (!flip)
        {
          fractions.push_back(a_neighborhood_geometry.getStoredMoments(k*9+j*3+i));
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
        }
        else
        {
          fractions.push_back(1 - a_neighborhood_geometry.getStoredMoments(k*9+j*3+i));
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          fractions.push_back((a_gas_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+0] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[0])/dx);
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+1] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[1])/dy);
          fractions.push_back((a_liquid_centroids[3*i+9*j+27*k+2] - a_neighborhood_geometry.getCell(k*9+j*3+i).calculateCentroid()[2])/dz);
        }
      }
    }
  }

  std::vector<double> center = sm.get_mass_centers_all(fractions);
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

  flag[0] = 0;
  n = t2.get_normal(fractions);
  n.normalize();

  switch (direction)
  {
    case 1:
      n[0] = -n[0];
      break;
    case 2:
      n[1] = -n[1];
      break;
    case 3:
      n[2] = -n[2];
      break;
    case 4:
      n[0] = -n[0];
      n[1] = -n[1];
      break;
    case 5:
      n[0] = -n[0];
      n[2] = -n[2];
      break;
    case 6:
      n[1] = -n[1];
      n[2] = -n[2];
      break;
    case 7:
      n[0] = -n[0];
      n[1] = -n[1];
      n[2] = -n[2];
      break;
  }

  if (!flip)
  {
    n[0] = -n[0];
    n[1] = -n[1];
    n[2] = -n[2];
  }

  previous = n;
  const IRL::Normal& n1 = n;
  const double d = a_neighborhood_geometry.getCenterCellStoredMoments();
  const IRL::RectangularCuboid& cube = a_neighborhood_geometry.getCenterCell();
  double distance = IRL::findDistanceOnePlane(cube, d, n1);
  return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, distance));
}

void loadML(std::string name/*, std::string name1, std::string name2*/)
{
  t.load_model(name, 0);
  //t2.load_model(name1, 1);
  //b.load_model(name2, 0);
}

void loadML2(std::string name/*, std::string name1, std::string name2*/)
{
  t2.load_model(name, 0);
  //t2.load_model(name1, 1);
  //b.load_model(name2, 0);
}

template <class CellType>
PlanarSeparator reconstructionWithLVIRA2D(
    const LVIRANeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  LVIRA_2D<CellType> lvira_system;
  return lvira_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
}

template <class CellType>
PlanarSeparator reconstructionWithLVIRA3D(
    const LVIRANeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  LVIRA_3D<CellType> lvira_system;
  return lvira_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
}

template <class CellType>
PlanarSeparator reconstructionWithMOF2D(
    const CellType& a_cell, const SeparatedMoments<VolumeMoments>& a_svm,
    const double a_internal_weight, const double a_external_weight) {
  MOF_2D<CellType> mof_solver;
  return mof_solver.solve(
      CellGroupedMoments<CellType, SeparatedMoments<VolumeMoments>>(&a_cell,
                                                                    &a_svm),
      a_internal_weight, a_external_weight);
}

template <class CellType>
PlanarSeparator reconstructionWithMOF3D(
    const CellType& a_cell, const SeparatedMoments<VolumeMoments>& a_svm,
    const double a_internal_weight, const double a_external_weight) {
  MOF_3D<CellType> mof_solver;
  return mof_solver.solve(
      CellGroupedMoments<CellType, SeparatedMoments<VolumeMoments>>(&a_cell,
                                                                    &a_svm),
      a_internal_weight, a_external_weight);
}

template <class MomentsContainerType, class CellType>
PlanarSeparator reconstructionWithAdvectedNormals(
    const MomentsContainerType& a_volume_moments_list,
    const R2PNeighborhood<CellType>& a_neighborhood,
    const double a_two_plane_threshold) {
  return AdvectedPlaneReconstruction::solve(
      a_volume_moments_list, a_neighborhood, a_two_plane_threshold);
}

//**********************************************************************
//     Function template definitions/inlined functions below this.
//     For debug versions.
//**********************************************************************

template <class CellType>
PlanarSeparator reconstructionWithR2P2DDebug(
    const R2PNeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  cleanReconstruction(
      a_neighborhood_geometry.getCenterCell(),
      (a_neighborhood_geometry.getCenterCellStoredMoments())[0].volume() /
          a_neighborhood_geometry.getCenterCell().calculateVolume(),
      &a_initial_reconstruction);
  if (a_initial_reconstruction.getNumberOfPlanes() == 1) {
    R2PDebug<R2P_2D1P<CellType>> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  } else {
    R2PDebug<R2P_2D2P<CellType>> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  }
}

template <class CellType>
PlanarSeparator reconstructionWithR2P3DDebug(
    const R2PNeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  cleanReconstruction(
      a_neighborhood_geometry.getCenterCell(),
      (a_neighborhood_geometry.getCenterCellStoredMoments())[0].volume() /
          a_neighborhood_geometry.getCenterCell().calculateVolume(),
      &a_initial_reconstruction);
  if (a_initial_reconstruction.getNumberOfPlanes() == 1) {
    R2PDebug<R2P_3D1P<CellType>> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  } else {
    R2PDebug<R2P_3D2P<CellType>> r2p_system;
    return r2p_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
  }
}

PlanarSeparator reconstructionWithELVIRA2DDebug(
    const ELVIRANeighborhood& a_neighborhood_geometry) {
  ELVIRADebug<ELVIRA_2D> elvira_system;
  return elvira_system.solve(&a_neighborhood_geometry);
}

PlanarSeparator reconstructionWithELVIRA3DDebug(
    const ELVIRANeighborhood& a_neighborhood_geometry) {
  ELVIRADebug<ELVIRA_3D> elvira_system;
  return elvira_system.solve(&a_neighborhood_geometry);
}

template <class CellType>
PlanarSeparator reconstructionWithLVIRA2DDebug(
    const LVIRANeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  LVIRADebug<LVIRA_2D<CellType>> lvira_system;
  return lvira_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
}

template <class CellType>
PlanarSeparator reconstructionWithLVIRA3DDebug(
    const LVIRANeighborhood<CellType>& a_neighborhood_geometry,
    PlanarSeparator a_initial_reconstruction) {
  LVIRADebug<LVIRA_3D<CellType>> lvira_system;
  return lvira_system.solve(a_neighborhood_geometry, a_initial_reconstruction);
}

template <class CellType>
PlanarSeparator reconstructionWithMOF2DDebug(
    const CellType& a_cell, const SeparatedMoments<VolumeMoments>& a_svm,
    const double a_internal_weight, const double a_external_weight) {
  MOFDebug<MOF_2D<CellType>> mof_solver;
  return mof_solver.solve(
      CellGroupedMoments<CellType, SeparatedMoments<VolumeMoments>>(&a_cell,
                                                                    &a_svm),
      a_internal_weight, a_external_weight);
}

template <class CellType>
PlanarSeparator reconstructionWithMOF3DDebug(
    const CellType& a_cell, const SeparatedMoments<VolumeMoments>& a_svm,
    const double a_internal_weight, const double a_external_weight) {
  MOFDebug<MOF_3D<CellType>> mof_solver;
  return mof_solver.solve(
      CellGroupedMoments<CellType, SeparatedMoments<VolumeMoments>>(&a_cell,
                                                                    &a_svm),
      a_internal_weight, a_external_weight);
}

template <class MomentsContainerType, class CellType>
PlanarSeparator reconstructionWithAdvectedNormalsDebug(
    const MomentsContainerType& a_volume_moments_list,
    const R2PNeighborhood<CellType>& a_neighborhood,
    const double a_two_plane_threshold) {
  return AdvectedPlaneReconstructionDebug::solve(
      a_volume_moments_list, a_neighborhood, a_two_plane_threshold);
}

}  // namespace IRL

#endif // IRL_INTERFACE_RECONSTRUCTION_METHODS_RECONSTRUCTION_INTERFACE_TPP_
