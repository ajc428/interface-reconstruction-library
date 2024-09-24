// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_SPATIAL_MOMENTS_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_SPATIAL_MOMENTS_H_

#include "mesh.h"
#include "data_mesh.h"
#include <math.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

using namespace std;

namespace IRL 
{
    class spatial_moments
    {
    private:
        double m000;
        double m100;
        double m010;
        double m001;
        double xc;
        double yc;
        double zc;
        double mu101;
        double mu011;
        double mu110;
        double mu200;
        double mu020;
        double mu002;
        double mu300;
        double mu030;
        double mu003;
        double mu210;
        double mu201;
        double mu120;
        double mu102;
        double mu021;
        double mu012;
        double mu111;

    public:
        spatial_moments();
        torch::Tensor calculate_moments(vector<double>, IRL::Normal, int);
        vector<double> get_mass_centers(vector<double> fractions);
        vector<double> get_mass_centers_all(vector<double>* fractions);
        vector<double> get_moment_of_intertia(vector<double>* fractions);

        vector<double> getMoments()
        {
            vector<double> temp;
            temp.push_back(mu200);
            temp.push_back(mu020);
            temp.push_back(mu002);
            temp.push_back(mu110);
            temp.push_back(mu101);
            temp.push_back(mu011);

            return temp;
        }
    };
}

#include "irl/machine_learning_reconstruction/spatial_moments.tpp"

#endif