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
        double m200;
        double m020;
        double m002;
        double xc;
        double yc;
        double zc;
        double mu100;
        double mu010;
        double mu001;
        double mu101;
        double mu011;
        double mu110;
        double mu111;
        double mu200;
        double mu020;
        double mu002;
        double mu210;
        double mu201;
        double mu120;
        double mu102;
        double mu021;
        double mu012;
        double mu300;
        double mu030;
        double mu003;
        torch::Tensor moments;

    public:
        spatial_moments();
        torch::Tensor calculate_moments(const DataMesh<double>&, Mesh);
        torch::Tensor getMoments();
    };
}

#include "irl/machine_learning_reconstruction/spatial_moments.tpp"

#endif