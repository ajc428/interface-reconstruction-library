// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_FRACTIONS_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_FRACTIONS_H_

#include <torch/torch.h>
#include <Eigen/Dense>

#include "irl/geometry/general/pt.h"
#include "irl/geometry/general/reference_frame.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/paraboloid_reconstruction/aligned_paraboloid.h"
#include "irl/paraboloid_reconstruction/paraboloid.h"
#include "irl/paraboloid_reconstruction/parametrized_surface.h"
#include "irl/geometry/general/pt_with_data.h"
#include "irl/geometry/half_edge_structures/half_edge_polyhedron_paraboloid.h"
#include "irl/geometry/half_edge_structures/segmented_half_edge_polyhedron_paraboloid.h"
#include "irl/generic_cutting/generic_cutting_definitions.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "irl/moments/volume_with_gradient.h"
#include "irl/paraboloid_reconstruction/gradient_paraboloid.h"

#include "mesh.h"
#include "data_mesh.h"

using namespace std;

namespace IRL 
{
    class fractions
    {
    private:
        Mesh mesh;
        int a_number_of_cells;
        std::array<double, 3> angles;
        std::array<vector<double>, 8> gradients;

        Mesh initializeMesh(const int); 
        bool isParaboloidInCenterCell(const IRL::Paraboloid&, const DataMesh<double>&);
    public:
        fractions(const int);

        IRL::Paraboloid new_parabaloid(double, double, double, double, double, double, double, double);
        IRL::Paraboloid new_random_parabaloid(double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double);
        torch::Tensor get_fractions(IRL::Paraboloid, bool);
        torch::Tensor get_fractions(IRL::Plane, bool);
        torch::Tensor get_fractions_with_gradients(IRL::Paraboloid, bool);
        torch::Tensor get_gradients(int);

        template <class MomentType, class SurfaceType>
        IRL::AddSurfaceOutput<MomentType, SurfaceType> getCellMomentsAndSurface(const IRL::Paraboloid&, const DataMesh<double>&, int, int, int); 
        
        template <class MomentType>
        MomentType getCellMoments(const IRL::Paraboloid&,const DataMesh<double>&, int, int, int); 

        template <class MomentType>
        MomentType getCellMoments(const IRL::Plane&,const DataMesh<double>&, int, int, int); 

        template <class MomentType, class SurfaceType>
        IRL::AddSurfaceOutput<MomentType, SurfaceType> getCellMomentsAndSurfaceWithGradients(const IRL::Paraboloid&, const DataMesh<double>&, int, int, int); 
        
        template <class MomentType>
        MomentType getCellMomentsWithGradients(const IRL::Paraboloid&,const DataMesh<double>&, int, int, int); 

        std::array<double, 3> getAngles()
        {
            return angles;
        }

        Mesh getMesh()
        {
            return mesh;
        }
    };
}

#include "irl/machine_learning_reconstruction/fractions.tpp"

#endif