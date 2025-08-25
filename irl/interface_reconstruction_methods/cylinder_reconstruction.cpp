// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/interface_reconstruction_methods/cylinder_reconstruction.h"

#include <cmath>
#include <iostream>

#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/interface_reconstruction_methods/progressive_radius_solver_cylinder.h"
#include "irl/parameters/constants.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace IRL {

Cylinder cylinder_reconstruction::solve(
    const cylinderNeighborhood* a_neighborhood_pointer) {
  assert(a_neighborhood_pointer != nullptr);
  neighborhood_VF_m = a_neighborhood_pointer;
  return this->solve();
}

Cylinder cylinder_reconstruction::solve(void) 
{
    IRL::Cylinder cylinder;
    Eigen::MatrixXd bary(27,3);
    Eigen::VectorXd VFs(27);
    int count = 0;
    int lim = 2;
    for (int ii = -lim; ii <= lim; ++ii) 
    {
        for (int jj = -lim; jj <= lim; ++jj) 
        {
            for (int kk = -lim; kk <= lim; ++kk) 
            {
                double VF1 = neighborhood_VF_m->getStoredMoments(ii,jj,kk).volume();
                if (VF1 > IRL::global_constants::VF_LOW && VF1 < IRL::global_constants::VF_HIGH)
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
    for (int ii = -lim; ii <= lim; ++ii) 
    {
        for (int jj = -lim; jj <= lim; ++jj) 
        {
            for (int kk = -lim; kk <= lim; ++kk) 
            {
                double VF1 = neighborhood_VF_m->getStoredMoments(ii,jj,kk).volume();
                if (VF1 > IRL::global_constants::VF_LOW && VF1 < IRL::global_constants::VF_HIGH)
                {
                    Pt a_liquid_centroid = neighborhood_VF_m->getStoredMoments(ii,jj,kk).centroid();
                    bary(count,0) = a_liquid_centroid[0];
                    bary(count,1) = a_liquid_centroid[1];
                    bary(count,2) = a_liquid_centroid[2];
                    VFs(count) = VF1;
                    //std::cout << VF1 << std::endl;
                    ++count;
                    datum = datum + a_liquid_centroid * VF1;
                    VF = VF + VF1;
                }
            }
        }
    }//std::cout << std::endl << std::endl;
    //std::cout << bary << std::endl << std::endl;
    datum = datum / VF;
    IRL::Pt temp = datum;

    PrincipalCurve pc = PrincipalCurve(bary, VFs);
    pc.fit();
    Eigen::MatrixXd curve = pc.getCurve();
    IRL::Normal direction = IRL::Normal(1.0,0.0,0.0);
    IRL::Pt center = neighborhood_VF_m->getCell(0,0,0).calculateCentroid();
    pc.fitSpline(&direction, &datum, IRL::Pt(center[0],center[1],center[2]));
//datum = temp;
    direction.normalize();
    double n3 = 0;
    double n2 = 0;
    double n1 = 0;
    IRL::Normal v1;
    if (abs(direction[0]) >= abs(direction[1]) && abs(direction[0]) >= abs(direction[2]))
    {
    n2 = direction[0]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
    n1 = (-n2*direction[1])/direction[0];
    v1[0] = n1; v1[1] = n2; v1[2] = 0;
    }
    else if (abs(direction[1]) >= abs(direction[0]) && abs(direction[1]) >= abs(direction[2]))
    {
    n1 = direction[1]/(sqrt(direction[1]*direction[1]+direction[0]*direction[0]));
    n2 = (-n1*direction[0])/direction[1];
    v1[0] = n1; v1[1] = n2; v1[2] = 0;
    }
    else if (abs(direction[2]) >= abs(direction[0]) && abs(direction[2]) >= abs(direction[1]))
    {
    n2 = direction[2]/(sqrt(direction[1]*direction[1]+direction[2]*direction[2]));
    n3 = (-n2*direction[1])/direction[2];
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
    
    const IRL::Pt lower_cell_pt(neighborhood_VF_m->getCell(0,0,0).calculateCentroid()[0]-neighborhood_VF_m->getCell(0,0,0).calculateSideLength(0)/2,
    neighborhood_VF_m->getCell(0,0,0).calculateCentroid()[1]-neighborhood_VF_m->getCell(0,0,0).calculateSideLength(1)/2, 
    neighborhood_VF_m->getCell(0,0,0).calculateCentroid()[2]-neighborhood_VF_m->getCell(0,0,0).calculateSideLength(2)/2);
    const IRL::Pt upper_cell_pt(neighborhood_VF_m->getCell(0,0,0).calculateCentroid()[0]+neighborhood_VF_m->getCell(0,0,0).calculateSideLength(0)/2,
    neighborhood_VF_m->getCell(0,0,0).calculateCentroid()[1]+neighborhood_VF_m->getCell(0,0,0).calculateSideLength(1)/2, 
    neighborhood_VF_m->getCell(0,0,0).calculateCentroid()[2]+neighborhood_VF_m->getCell(0,0,0).calculateSideLength(2)/2);

    auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt, upper_cell_pt);
    IRL::ProgressiveRadiusSolverCylinder<IRL::RectangularCuboid>
    solver_radius(cell, neighborhood_VF_m->getStoredMoments(0,0,0).volume(), 1.0e-14,
    cylinder);

    cylinder = solver_radius.getCylinder();
    //std::cout << cylinder << std::endl << std::endl;
    //std::cout << direction << std::endl << std::endl;

  return cylinder;
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
        new_curve.row(idx) += data.row(i)*VF(i)*VF(i);
        counts(idx)++;
        VF_total(idx) +=  VF(i)*VF(i);
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

}  // namespace IRL
