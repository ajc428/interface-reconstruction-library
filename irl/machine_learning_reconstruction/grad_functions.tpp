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
    grad_functions::grad_functions(int num_cells)
    {
        gen = new IRL::fractions(num_cells);
    }

    grad_functions::~grad_functions()
    {
        delete gen;
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
            torch::Tensor temp = torch::zeros(108);
            for (int j = 0; j < 108; ++j)
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
}

#endif