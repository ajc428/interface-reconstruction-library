// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_GRAD_FUNCTIONS_TPP_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_GRAD_FUNCTIONS_TPP_

using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;

using torch::autograd::variable_list;
using torch::autograd::tensor_list;

using torch::autograd::compute_requires_grad;
using torch::autograd::collect_next_edges;
using torch::autograd::flatten_tensor_args;

namespace IRL
{
    grad_functions::grad_functions(int num_cells, int s)
    {
        gen = new IRL::fractions(num_cells);
        sm = new IRL::spatial_moments();
        if (s == 3)
        {
            size = 3;
        }
        else
        {
            size = 189;
        }
    }

    grad_functions::~grad_functions()
    {
        delete gen;
        delete sm;
    }

    torch::Tensor grad_functions::LVIRAForward(const torch::Tensor y_pred, const torch::Tensor fractions) 
    {
        IRL::PlanarSeparator a_interface;

        IRL::LVIRANeighborhood<IRL::RectangularCuboid> neighborhood;
        neighborhood.resize(27);
        neighborhood.setCenterOfStencil(13);
        IRL::RectangularCuboid cells[27];
        for (int i = 0; i < 3; ++i) 
        {
            for (int j = 0; j < 3; ++j) 
            {
                for (int k = 0; k < 3; ++k) 
                {
                    double* a = new double();
                    *a = fractions[7*(i*9+j*3+k)].item<double>();
                    const double* b = a;
                    const int local_index = k*9 + j*3 + i;
                    cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
                        IRL::Pt(i-1.5, j-1.5, k-1.5),
                        IRL::Pt(i-0.5, j-0.5, k-0.5));
                    neighborhood.setMember(
                        static_cast<IRL::UnsignedIndex_t>(local_index),
                        &cells[local_index], b);
                }
            }
        }

        IRL::Pt a_gas_centroid = IRL::Pt(fractions[7*(1*9+1*3+1)+4].item<double>(), fractions[7*(1*9+1*3+1)+5].item<double>(), fractions[7*(1*9+1*3+1)+6].item<double>());
        IRL::Pt a_liquid_centroid = IRL::Pt(fractions[7*(1*9+1*3+1)+1].item<double>(), fractions[7*(1*9+1*3+1)+2].item<double>(), fractions[7*(1*9+1*3+1)+3].item<double>());
        auto bary_normal = IRL::Normal::fromPtNormalized(
            a_gas_centroid - a_liquid_centroid);
        bary_normal.normalize();
        const double initial_distance =
            bary_normal * neighborhood.getCenterCell().calculateCentroid();
        a_interface = IRL::PlanarSeparator::fromOnePlane(
            IRL::Plane(bary_normal, initial_distance));
        IRL::setDistanceToMatchVolumeFractionPartialFill(
            neighborhood.getCenterCell(),
            neighborhood.getCenterCellStoredMoments(),
            &a_interface);

        double e = 0.01;//std::sqrt(DBL_EPSILON);
        torch::Tensor weights = torch::zeros({27,27});
        double* w = new double[27];

        for (int i = 0; i < 27; ++i)
        {
            w[i] = (y_pred[i].item<double>());
            weights.index_put_({i,torch::indexing::Slice()},y_pred);
            weights[i][i] += e;
        }
        a_interface = IRL::reconstructionWithLVIRA3D(neighborhood,a_interface,w);
        torch::Tensor result = torch::zeros({3,1});
        result[0] = -a_interface[0].normal()[0];
        result[1] = -a_interface[0].normal()[1];
        result[2] = -a_interface[0].normal()[2];

        vector<torch::Tensor> ep;
        vector<torch::Tensor> grads;
        torch::Tensor result_grad = torch::zeros({3,1});

        for (int i = 0; i < 27; ++i)
        {
            for (int j = 0; j < 27; ++j)
            {
                w[i] = weights[i][j].item<double>();
            }
            a_interface = IRL::reconstructionWithLVIRA3D(neighborhood,a_interface,w);
            result_grad[0] = -a_interface[0].normal()[0];
            result_grad[1] = -a_interface[0].normal()[1];
            result_grad[2] = -a_interface[0].normal()[2];
            ep.push_back(result_grad);
            torch::Tensor temp = torch::zeros({3,1});
            for (int j = 0; j < 3; ++j)
            {
                temp[j] = (ep[i][j].item<double>() - result[j].item<double>()) / e;
            }
            grads.push_back(temp);
        }
        if (compute_requires_grad(y_pred)) 
        {
            auto grad_fn = std::shared_ptr<ELVIRABackward>(new grad_functions::ELVIRABackward(), deleteNode);
            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->ELVIRA_grads = grads;
            grad_fn->y_pred = y_pred;
            set_history(flatten_tensor_args(result), grad_fn);
        }
        delete w;
        return result;
    }

    torch::Tensor grad_functions::ELVIRAForward(const torch::Tensor y_pred, const torch::Tensor fractions) 
    {
        IRL::PlanarSeparator a_interface;

        IRL::ELVIRANeighborhood neighborhood;
        neighborhood.resize(27);
        IRL::RectangularCuboid cells[27];
        for (int i = 0; i < 3; ++i) 
        {
            for (int j = 0; j < 3; ++j) 
            {
                for (int k = 0; k < 3; ++k) 
                {
                    double* a = new double();
                    *a = fractions[7*(i*9+j*3+k)].item<double>();
                    const double* b = a;
                    const int local_index = k*9 + j*3 + i;
                    cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
                        IRL::Pt(i-1.5, j-1.5, k-1.5),
                        IRL::Pt(i-0.5, j-0.5, k-0.5));
                    neighborhood.setMember(&cells[local_index],b,i-1,j-1,k-1);
                }
            }
        }


        double e = 0.01;//std::sqrt(DBL_EPSILON);
        torch::Tensor weights = torch::zeros({27,27});
        double* w = new double[27];

        for (int i = 0; i < 27; ++i)
        {
            w[i] = y_pred[i].item<double>();
            weights.index_put_({i,torch::indexing::Slice()},y_pred);
            weights[i][i] += e;
        }
        a_interface = IRL::reconstructionWithELVIRA3D(neighborhood,w);
        torch::Tensor result = torch::zeros({3,1});
        result[0] = -a_interface[0].normal()[0];
        result[1] = -a_interface[0].normal()[1];
        result[2] = -a_interface[0].normal()[2];

        vector<torch::Tensor> ep;
        vector<torch::Tensor> grads;
        torch::Tensor result_grad = torch::zeros({3,1});

        for (int i = 0; i < 27; ++i)
        {
            for (int j = 0; j < 27; ++j)
            {
                w[i] = weights[i][j].item<double>();
            }
            a_interface = IRL::reconstructionWithELVIRA3D(neighborhood,w);
            result_grad[0] = -a_interface[0].normal()[0];
            result_grad[1] = -a_interface[0].normal()[1];
            result_grad[2] = -a_interface[0].normal()[2];
            ep.push_back(result_grad);
            torch::Tensor temp = torch::zeros({3,1});
            for (int j = 0; j < 3; ++j)
            {
                temp[j] = (ep[i][j].item<double>() - result[j].item<double>()) / e;
            }
            grads.push_back(temp);
        }
        if (compute_requires_grad(y_pred)) 
        {
            auto grad_fn = std::shared_ptr<ELVIRABackward>(new grad_functions::ELVIRABackward(), deleteNode);
            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->ELVIRA_grads = grads;
            grad_fn->y_pred = y_pred;
            set_history(flatten_tensor_args(result), grad_fn);
        }
        delete w;
        return result;
    }

    torch::Tensor grad_functions::VolumeFracsForward(const torch::Tensor y_pred) 
    {
        IRL::Paraboloid paraboloid;
        paraboloid = gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>());

        auto result = gen->get_fractions_with_gradients(paraboloid, true);
        vector<torch::Tensor> grads;
        for (int i = 0; i < 8; ++i)
        {
            grads.push_back(gen->get_gradients(i));
        }

        if (compute_requires_grad(y_pred)) 
        {
            auto grad_fn = std::shared_ptr<VolumeFracsBackward>(new grad_functions::VolumeFracsBackward(), deleteNode);

            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->frac_grads = grads;
            grad_fn->y_pred = y_pred;

            set_history(flatten_tensor_args(result), grad_fn);
        }
        return result;
    }

    torch::Tensor grad_functions::VolumeFracsForwardFD(const torch::Tensor y_pred) 
    {
        IRL::Paraboloid paraboloid = gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>());

        vector<IRL::Paraboloid> p;
        vector<IRL::Paraboloid> p1;
        double e = std::sqrt(DBL_EPSILON);

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>()+e, y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>()));

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>()+e, y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>()));

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>()+e,
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>()));

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>()+e, y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>()));

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>()+e, y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>()));

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>()+e,
        y_pred[6].item<double>(), y_pred[7].item<double>()));

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>()+e, y_pred[7].item<double>()));

        p.push_back(gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>()+e));

        auto result = gen->get_fractions(paraboloid, true);

        vector<torch::Tensor> ep;
        vector<torch::Tensor> grads;

        for (int i = 0; i < 8; ++i)
        {
            ep.push_back(gen->get_fractions(p[i], true));
            torch::Tensor temp = torch::zeros(size);
            for (int j = 0; j < size; ++j)
            {
                temp[j] = (ep[i][j].item<double>() - result[j].item<double>()) / e;
            }
            grads.push_back(temp);
        }
        if (compute_requires_grad(y_pred)) 
        {
            auto grad_fn = std::shared_ptr<VolumeFracsBackward>(new grad_functions::VolumeFracsBackward(), deleteNode);

            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->frac_grads = grads;
            grad_fn->y_pred = y_pred;

            set_history(flatten_tensor_args(result), grad_fn);
        }
        return result;
    }

    torch::Tensor grad_functions::R2PVolumeFracsForward(const torch::Tensor y_pred,double vf,double d1_o,double d2_o) 
    {
        vector<double> fractions;
        auto n = IRL::Normal();
        auto n1 = IRL::Normal();
        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        n1[0] = y_pred[3].item<double>();
        n1[1] = y_pred[4].item<double>();
        n1[2] = y_pred[5].item<double>();
        n.normalize();
        n1.normalize();
        /*n[0] = sin(y_pred[1].item<double>()) * cos(y_pred[0].item<double>());
        n[1] = sin(y_pred[1].item<double>()) * sin(y_pred[0].item<double>());
        n[2] = cos(y_pred[1].item<double>());
        n1[0] = sin(y_pred[3].item<double>()) * cos(y_pred[2].item<double>());
        n1[1] = sin(y_pred[3].item<double>()) * sin(y_pred[2].item<double>());
        n1[2] = cos(y_pred[3].item<double>());*/
        auto plane = IRL::Plane(n, d1_o);
        auto plane1 = IRL::Plane(n1, d2_o);
        auto planes = IRL::PlanarSeparator::fromTwoPlanes(plane,plane1,1);

        vector<IRL::PlanarSeparator> p;
        double e = std::sqrt(DBL_EPSILON);

        n[0] = n[0] + e;
        plane = IRL::Plane(n, d1_o);
        plane1 = IRL::Plane(n1, d2_o);
        auto p1 = IRL::PlanarSeparator::fromTwoPlanes(plane,plane1,1);
        p.push_back(p1);

        n[0] = n[0] - e;
        n[1] = n[1] + e;
        plane = IRL::Plane(n, d1_o);
        plane1 = IRL::Plane(n1, d2_o);
        p1 = IRL::PlanarSeparator::fromTwoPlanes(plane,plane1,1);
        p.push_back(p1);

        n[1] = n[1] - e;
        n[2] = n[2] + e;
        plane = IRL::Plane(n, d1_o);
        plane1 = IRL::Plane(n1, d2_o);
        p1 = IRL::PlanarSeparator::fromTwoPlanes(plane,plane1,1);
        p.push_back(p1);

        n[2] = n[2] - e;
        n1[0] = n1[0] + e;
        plane = IRL::Plane(n, d1_o);
        plane1 = IRL::Plane(n1, d2_o);
        p1 = IRL::PlanarSeparator::fromTwoPlanes(plane,plane1,1);
        p.push_back(p1);

        n1[0] = n1[0] - e;
        n1[1] = n1[1] + e;
        plane = IRL::Plane(n, d1_o);
        plane1 = IRL::Plane(n1, d2_o);
        p1 = IRL::PlanarSeparator::fromTwoPlanes(plane,plane1,1);
        p.push_back(p1);

        n1[1] = n1[1] - e;
        n1[2] = n1[2] + e;
        plane = IRL::Plane(n, d1_o);
        plane1 = IRL::Plane(n1, d2_o);
        p1 = IRL::PlanarSeparator::fromTwoPlanes(plane,plane1,1);
        p.push_back(p1);

        auto cell = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(-0.5,-0.5,-0.5),IRL::Pt(0.5,0.5,0.5));
        const IRL::RectangularCuboid a_cell = cell;
        const double v = vf;
        IRL::setDistanceToMatchVolumeFractionPartialFill<IRL::PlanarSeparator>(a_cell,v,&planes);

        auto result = gen->get_fractions_all(planes);
        vector<torch::Tensor> ep;
        vector<torch::Tensor> grads;
        for (int i = 0; i < 6; ++i)
        {
            IRL::setDistanceToMatchVolumeFractionPartialFill<IRL::PlanarSeparator>(a_cell,v,&p[i]);
            ep.push_back(gen->get_fractions_all(p[i]));
            torch::Tensor temp = torch::zeros(size);
            temp = (ep[i] - result) / e;
            grads.push_back(temp);
        }

        //torch::Tensor grads2 = torch::zeros({4,6});
        /*grads2[0][0] = -sin(y_pred[1].item<double>()) * sin(y_pred[0].item<double>());
        grads2[0][1] = sin(y_pred[1].item<double>()) * cos(y_pred[0].item<double>());
        grads2[1][0] = cos(y_pred[1].item<double>()) * cos(y_pred[0].item<double>());
        grads2[1][1] = cos(y_pred[1].item<double>()) * sin(y_pred[0].item<double>());
        grads2[1][2] = -sin(y_pred[1].item<double>());
        grads2[2][3] = -sin(y_pred[3].item<double>()) * sin(y_pred[2].item<double>());
        grads2[2][4] = sin(y_pred[3].item<double>()) * cos(y_pred[2].item<double>());;
        grads2[3][3] = cos(y_pred[3].item<double>()) * cos(y_pred[2].item<double>());
        grads2[3][4] = cos(y_pred[3].item<double>()) * sin(y_pred[2].item<double>());
        grads2[3][5] = -sin(y_pred[3].item<double>());

        torch::Tensor grads3 = torch::matmul(grads2,grads1);
        vector<torch::Tensor> grads;
        for (int i = 0; i < 4; ++i)
        {
            grads.push_back(grads3[i]);
        }*/

        if (compute_requires_grad(y_pred)) 
        {
            auto grad_fn = std::shared_ptr<R2PBackward>(new grad_functions::R2PBackward(), deleteNode);

            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->R2P_grads = grads;
            grad_fn->y_pred = y_pred;

            set_history(flatten_tensor_args(result), grad_fn);
        }
        return result;
    }
}

#endif