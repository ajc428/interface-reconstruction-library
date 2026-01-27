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
    const cylinderNeighborhood* a_neighborhood_pointer, const int f) {
  assert(a_neighborhood_pointer != nullptr);
  neighborhood_VF_m = a_neighborhood_pointer;
  flip = f;
  return this->solve();
}

Cylinder cylinder_reconstruction::solve(void) 
{
    IRL::Cylinder cylinder;
    std::vector<IRL::Pt> bary_vector;
    std::vector<double> VF_vector;
    int lim = 2;
    for (int ii = -lim; ii <= lim; ++ii) 
    {
        for (int jj = -lim; jj <= lim; ++jj) 
        {
            for (int kk = -lim; kk <= lim; ++kk) 
            {
                auto moments = neighborhood_VF_m->getStoredMoments(ii, jj, kk);
                double VF1 = moments.volume();
                if (VF1 > IRL::global_constants::VF_LOW && VF1 < IRL::global_constants::VF_HIGH)
                {
                    bary_vector.push_back(moments.centroid());
                    VF_vector.push_back(VF1);
                }
            }
        }
    }

    if (VF_vector.size() < 3)
    {
        cylinder = IRL::Cylinder(bary_vector[0], IRL::ReferenceFrame(IRL::Normal(1.0,0.0,0.0),IRL::Normal(0.0,1.0,0.0),IRL::Normal(0.0,0.0,1.0)), 1, 0.0, flip);
    }
    else
    {
        Eigen::MatrixXd bary(bary_vector.size(), 3);
        Eigen::VectorXd VFs(VF_vector.size());
        for (size_t i = 0; i < bary_vector.size(); ++i) 
        {
            bary.row(i) = Eigen::RowVector3d(bary_vector[i][0], bary_vector[i][1], bary_vector[i][2]);
            VFs(i) = VF_vector[i];
        }

        const auto& central_cell = neighborhood_VF_m->getCell(0, 0, 0);
        const IRL::Pt cell_centroid = central_cell.calculateCentroid();
        const IRL::Pt cell_side = IRL::Pt(central_cell.calculateSideLength(0)/2,central_cell.calculateSideLength(1)/2,central_cell.calculateSideLength(2)/2);
        const IRL::Pt lower_cell_pt(cell_centroid[0]-cell_side[0],cell_centroid[1]-cell_side[1],cell_centroid[2]-cell_side[2]);
        const IRL::Pt upper_cell_pt(cell_centroid[0]+cell_side[0],cell_centroid[1]+cell_side[1],cell_centroid[2]+cell_side[2]);

        PrincipalCurve pc = PrincipalCurve(bary, VFs, central_cell.calculateSideLength(0));
        pc.fit();
        Eigen::MatrixXd curve = pc.getCurve();

        IRL::Normal direction = IRL::Normal(1.0,0.0,0.0);
        IRL::Pt center = neighborhood_VF_m->getCell(0,0,0).calculateCentroid();
        IRL::Pt datum = IRL::Pt(0,0,0);
        pc.fitPoly(&direction, &datum, center);

        direction.normalize();
        IRL::Normal temp;
        if (direction[2] <= 0.97) 
        {
            temp = IRL::Normal(0, 0, 1);
        }
        else
        {
            temp = IRL::Normal(0, 1, 0);
        }
        IRL::Normal b = IRL::crossProduct(direction, temp);
        b.normalize();
        IRL::Normal a = IRL::crossProduct(b, direction);
        IRL::ReferenceFrame frame = IRL::ReferenceFrame(direction, a, b);

        cylinder = IRL::Cylinder(datum, frame, 1, 0.00025, flip);

        auto cell = IRL::RectangularCuboid::fromBoundingPts(lower_cell_pt, upper_cell_pt);
        double vf_target = neighborhood_VF_m->getStoredMoments(0,0,0).volume();
        if (flip < 0)
        {
            vf_target = 1.0 - vf_target;
        }

        IRL::ProgressiveRadiusSolverCylinder<IRL::RectangularCuboid>
        solver_radius(cell, vf_target, 1.0e-14,
        cylinder);

        cylinder = solver_radius.getCylinder();
    }

    return cylinder;
}



PrincipalCurve::PrincipalCurve(const Eigen::MatrixXd& d, const Eigen::VectorXd& VFs, const double d_x)
{
  data = d;
  VF = VFs;
  dx = d_x;
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

        double change = (curve - old_curve).rowwise().squaredNorm().maxCoeff() / (dx*dx);
        if (change < tolerance) 
        {
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
}

void PrincipalCurve::projectDataOntoCurve() 
{
    projection_indices.resize(data.rows());
    for (int i = 0; i < data.rows(); ++i) 
    {
        double min_dist_sq = std::numeric_limits<double>::max();
        std::vector<int> best_idx;

        for (int j = 0; j < curve.rows(); ++j) 
        {
            double dist_sq = (data.row(i) - curve.row(j)).squaredNorm();
            if (dist_sq < min_dist_sq-tol*dx) 
            {
                min_dist_sq = dist_sq;
                best_idx.clear();
                best_idx.push_back(j);
            }
            else if (dist_sq < min_dist_sq+tol*dx)
            {
                best_idx.push_back(j);
            }
        }
        projection_indices[i] = best_idx;
    }
}

void PrincipalCurve::updateCurve() 
{
    Eigen::MatrixXd new_curve = Eigen::MatrixXd::Zero(curve.rows(), curve.cols());
    Eigen::VectorXd VF_total = Eigen::VectorXd::Zero(curve.rows());
    for (int i = 0; i < data.rows(); ++i) 
    {
        std::vector<int> idx = projection_indices[i];
        double weight = 1.0/idx.size();
        for (int j = 0; j < idx.size(); ++j)
        {
            new_curve.row(idx[j]) += data.row(i)*VF(i)*weight;
            VF_total(idx[j]) +=  VF(i)*weight;
        }
    }

    for (int i = 0; i < curve.rows(); ++i) 
    {
        if (VF_total(i) > IRL::global_constants::VF_LOW) 
        {
            curve.row(i) = new_curve.row(i) / VF_total(i);
        }
    }
}

void PrincipalCurve::fitPoly(IRL::Normal *direction, IRL::Pt *pt, IRL::Pt target)
{
    Eigen::MatrixXd points = curve; 
    Eigen::VectorXd t(curve.rows());
    lambda.resize(curve.rows());
    lambda(0) = 0.0;
    for (int i = 1; i < curve.rows(); ++i) 
    {
        lambda(i) = lambda(i - 1) + (curve.row(i) - curve.row(i - 1)).norm();
    }
    for (int i = 0; i < curve.rows(); ++i)
    {
      t(i) = lambda(i) / lambda(curve.rows()-1);
    }
    double par = 0;
    double mag = 100;

    const double d0 = (t(0) - t(1)) * (t(0) - t(2));
    const double d1 = (t(1) - t(0)) * (t(1) - t(2));
    const double d2 = (t(2) - t(0)) * (t(2) - t(1));
    if (std::abs(d0) > IRL::global_constants::VF_LOW && std::abs(d1) > IRL::global_constants::VF_LOW)
    {
        const Eigen::Vector3d A = points.row(0) / d0 + points.row(1) / d1 + points.row(2) / d2;
        const Eigen::Vector3d B = -points.row(0) * (t(1) + t(2)) / d0 - points.row(1) * (t(0) + t(2)) / d1 - points.row(2) * (t(0) + t(1)) / d2;
        const Eigen::Vector3d C = points.row(0) * (t(1) * t(2)) / d0 + points.row(1) * (t(0) * t(2)) / d1 + points.row(2) * (t(0) * t(1)) / d2;
        const Eigen::Vector3d T(target[0], target[1], target[2]);
        const double a = 2.0 * A.dot(A);
        const double b = 3.0 * A.dot(B);
        const double c = 2.0 * A.dot(C-T) + B.dot(B);
        const double d = B.dot(C-T);
        std::vector<double> candidates = solveCubic(a, b, c, d);
        candidates.push_back(t(0));
        candidates.push_back(t(2));

        for (int i = 0; i < candidates.size(); ++i)
        {
            double t_cand = candidates[i];
            if (t_cand < t(0) || t_cand > t(2)) 
            {
                continue;
            }
            Eigen::Vector3d P_cand = A * t_cand * t_cand + B * t_cand + C;
            double mag2 = (P_cand - T).squaredNorm();
            if (mag2 < mag)
            {
                mag = mag2;
                par = t_cand;
            }
        }
        Eigen::Vector3d P = A * par * par + B * par + C;
        pt[0][0] = P(0);
        pt[0][1] = P(1);
        pt[0][2] = P(2);

        Eigen::Vector3d D = 2 * A * par + B;
        direction[0][0] = D(0);
        direction[0][1] = D(1);
        direction[0][2] = D(2);
    }
    else
    {
        pt[0][0] = curve.row(0)[0];
        pt[0][1] = curve.row(0)[1];
        pt[0][2] = curve.row(0)[2];

        direction[0][0] = 1.0;
        direction[0][1] = 0.0;
        direction[0][2] = 0.0;
    }
}

std::vector<double> PrincipalCurve::solveCubic(double a, double b, double c, double d) 
{
    constexpr double ep = 1e-12;

    if (std::abs(a) < ep) 
    {
        if (std::abs(b) < ep) 
        {
            if (std::abs(c) < ep) return {};
            return {-d / c};
        }
        double delta = c * c - 4.0 * b * d;
        if (delta > -ep)
        {
            double sqrt_delta = std::sqrt(std::max(delta,0.0));
            return {(-c + sqrt_delta) / (2.0 * b), (-c - sqrt_delta) / (2.0 * b)};
        }
        return {};
    }
    const double p = (3.0 * a * c - b * b) / (3.0 * a * a);
    const double q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    const double offset = -b / (3.0 * a);
    double discriminant = std::pow(q / 2.0, 2) + std::pow(p / 3.0, 3);
    std::vector<double> roots;

    if (discriminant > ep) 
    {
        const double sqrt_d = std::sqrt(discriminant);
        const double u = std::cbrt(-q / 2.0 + sqrt_d);
        const double v = (std::abs(u) > ep) ? -p / (3.0 * u) : 0.0;
        roots.push_back(u + v);

    } 
    else 
    {
        const double m = 2.0 * std::sqrt(-p / 3.0);
        const double phi = std::acos(std::max(-1.0, std::min(1.0, -q / (2.0 * std::sqrt(-std::pow(p / 3.0, 3))))));
        roots.push_back(m * std::cos(phi / 3.0));
        roots.push_back(m * std::cos((phi + 2.0 * M_PI) / 3.0));
        roots.push_back(m * std::cos((phi + 4.0 * M_PI) / 3.0));
    }
    
    for (double& root : roots) 
    {
        root += offset;
    }
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    return roots;
}

}  // namespace IRL
