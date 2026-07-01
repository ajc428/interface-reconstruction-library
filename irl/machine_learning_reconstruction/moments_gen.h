// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_MOMENTS_GEN_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_MOMENTS_GEN_H_

#include <cmath>
#include <random>
#include <torch/torch.h>
#include <Eigen/Dense>

#include "irl/geometry/general/pt.h"
#include "irl/geometry/general/reference_frame.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/paraboloid_reconstruction/paraboloid.h"
#include "irl/generic_cutting/generic_cutting_definitions.h"
#include "irl/generic_cutting/generic_cutting.h"

#include "stencil.h"

namespace IRL 
{
    class moments_gen
    {
    private:
        std::array<double, 3> angles;
        IRL::stencil *mesh;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution1;
        std::normal_distribution<double> distribution2;

        bool isParaboloidInCenterCell(IRL::Paraboloid);
    public:
        moments_gen(int, int, int, int, int, int, double, double, double);
        ~moments_gen();

        IRL::Paraboloid new_paraboloid(double, double, double, double, double, double, double, double);
        IRL::Paraboloid new_random_paraboloid(double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double);
 
        std::vector<double> get_moments(IRL::Paraboloid, int, bool, bool&);

        IRL::SeparatedMoments<IRL::VolumeMoments> getCellMoments(IRL::Paraboloid, int, int, int); 

        IRL::stencil* getStencil() {return mesh;};

        std::array<double, 3> getAngles() {return angles;};
    };
}

#endif