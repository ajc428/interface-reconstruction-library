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

        double mx000;
        double my000;
        double mz000;
        double mx100;
        double my100;
        double mz100;
        double mx010;
        double my010;
        double mz010;
        double mx001;
        double my001;
        double mz001;
        double x_xc;
        double x_yc;
        double x_zc;
        double y_xc;
        double y_yc;
        double y_zc;
        double z_xc;
        double z_yc;
        double z_zc;

        vector<double> centroid_x;
        vector<double> centroid_y;
        vector<double> centroid_z;

    public:
        spatial_moments();
        torch::Tensor calculate_moments(const DataMesh<double>&, DataMesh<IRL::Pt>&, Mesh);
        vector<double> get_mass_centers(vector<double> fractions);
        vector<double> get_mass_centers_all(vector<double> fractions);

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