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
            size = 12;
        }
        else if (s == 4 || s == 5)
        {
            size = 27;
        }
        else
        {
            size = 108;
        }
    }

    grad_functions::~grad_functions()
    {
        delete gen;
        delete sm;
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

    torch::Tensor grad_functions::MomentsForward(const torch::Tensor y_pred, DataMesh<double>& liquid_volume_fraction, DataMesh<IRL::Pt>& liquid_centroid) 
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

        auto fracs = gen->get_fractions(paraboloid, true);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    liquid_volume_fraction(i, j, k) = fracs[4*(i*9+j*3+k)].item<double>();
                    liquid_centroid(i, j, k) = IRL::Pt(fracs[4*(i*9+j*3+k)+1].item<double>(), fracs[4*(i*9+j*3+k)+2].item<double>(), fracs[4*(i*9+j*3+k)+3].item<double>());
                }
            }
        }
        auto result = sm->calculate_moments(liquid_volume_fraction, liquid_centroid, gen->getMesh());

        vector<torch::Tensor> ep;
        vector<torch::Tensor> grads;
        
        for (int n = 0; n < 8; ++n)
        {
            fracs = gen->get_fractions(p[n], true);
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        liquid_volume_fraction(i, j, k) = fracs[4*(i*9+j*3+k)].item<double>();
                        liquid_centroid(i, j, k) = IRL::Pt(fracs[4*(i*9+j*3+k)+1].item<double>(), fracs[4*(i*9+j*3+k)+2].item<double>(), fracs[4*(i*9+j*3+k)+3].item<double>());
                    }
                }
            }
            ep.push_back(sm->calculate_moments(liquid_volume_fraction, liquid_centroid, gen->getMesh()));
            torch::Tensor temp = torch::zeros(size);
            for (int j = 0; j < size; ++j)
            {
                temp[j] = (ep[n][j] - result[j]) / e;
            }
            grads.push_back(temp);
        }
        if (compute_requires_grad(y_pred)) 
        {
            auto grad_fn = std::shared_ptr<MomentsBackward>(new grad_functions::MomentsBackward(), deleteNode);

            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->moment_grads = grads;
            grad_fn->y_pred = y_pred;

            set_history(flatten_tensor_args(result), grad_fn);
        }
        return result;
    }

    torch::Tensor grad_functions::PLICForward(const torch::Tensor y_pred) 
    {
        vector<double> fractions;
        auto n = IRL::Normal();
        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        n.normalize();
        auto plane = IRL::Plane(n, y_pred[3].item<double>());

        vector<IRL::Plane> p;
        double e = std::sqrt(DBL_EPSILON);

        n[0] = y_pred[0].item<double>() + e;
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        n.normalize();
        p.push_back(IRL::Plane(n, y_pred[3].item<double>()));

        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>() + e;
        n[2] = y_pred[2].item<double>();
        n.normalize();
        p.push_back(IRL::Plane(n, y_pred[3].item<double>()));

        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>() + e;
        n.normalize();
        p.push_back(IRL::Plane(n, y_pred[3].item<double>()));

        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        n.normalize();
        p.push_back(IRL::Plane(n, y_pred[3].item<double>()+e));

        auto result = gen->get_fractions(plane, true);

        vector<torch::Tensor> ep;
        vector<torch::Tensor> grads;

        for (int i = 0; i < 4; ++i)
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
            auto grad_fn = std::shared_ptr<PLICBackward>(new grad_functions::PLICBackward(), deleteNode);

            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->PLIC_grads = grads;
            grad_fn->y_pred = y_pred;

            set_history(flatten_tensor_args(result), grad_fn);
        }
        return result;
    }

    torch::Tensor grad_functions::VolumeFracsNormalForward(const torch::Tensor y_pred, IRL::Normal norm) 
    {
        IRL::Pt x_dir = IRL::Pt(0,0,0);
        if (abs(norm[0]) >= abs(norm[1]) && abs(norm[1]) >= abs(norm[2]))
        {
            x_dir[0] = norm[1];
            x_dir[1] = -norm[0];
            x_dir[2] = 0;
        }
        else if (abs(norm[1]) >= abs(norm[0]) && abs(norm[0]) >= abs(norm[2]))
        {
            x_dir[0] = -norm[1];
            x_dir[1] = norm[0];
            x_dir[2] = 0;
        }
        else if (abs(norm[0]) >= abs(norm[2]) && abs(norm[2]) >= abs(norm[1]))
        {
            x_dir[0] = norm[2];
            x_dir[2] = -norm[0];
            x_dir[1] = 0;
        }
        else if (abs(norm[1]) >= abs(norm[2]) && abs(norm[2]) >= abs(norm[0]))
        {
            x_dir[0] = 0;
            x_dir[1] = norm[2];
            x_dir[2] = -norm[1];
        }
        else if (abs(norm[2]) >= abs(norm[0]) && abs(norm[0]) >= abs(norm[1]))
        {
            x_dir[0] = -norm[2];
            x_dir[2] = norm[0];
            x_dir[1] = 0;
        }
        else if (abs(norm[2]) >= abs(norm[1]) && abs(norm[1]) >= abs(norm[0]))
        {
            x_dir[0] = 0;
            x_dir[1] = -norm[2];
            x_dir[2] = norm[1];
        }
        IRL::Pt y_dir = IRL::Pt(0,0,0);
        y_dir[0] = norm[1] * x_dir[2] - norm[2] * x_dir[1];
        y_dir[1] = -(norm[0] * x_dir[2] - norm[2] * x_dir[0]);
        y_dir[2] = norm[0] * x_dir[1] - norm[1] * x_dir[0];

        IRL::Pt temp = x_dir;
        x_dir[0] = cos(y_pred[3].item<double>()) * temp[0] + sin(y_pred[3].item<double>()) * y_dir[0];
        x_dir[1] = cos(y_pred[3].item<double>()) * temp[1] + sin(y_pred[3].item<double>()) * y_dir[1];
        x_dir[2] = cos(y_pred[3].item<double>()) * temp[2] + sin(y_pred[3].item<double>()) * y_dir[2];

        y_dir[0] = norm[1] * x_dir[2] - norm[2] * x_dir[1];
        y_dir[1] = -(norm[0] * x_dir[2] - norm[2] * x_dir[0]);
        y_dir[2] = norm[0] * x_dir[1] - norm[1] * x_dir[0];

        IRL::Pt datum = IRL::Pt(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>());
        IRL::ReferenceFrame frame = IRL::ReferenceFrame(IRL::Normal(x_dir[0], x_dir[1], x_dir[2]), IRL::Normal(y_dir[0], y_dir[1], y_dir[2]), IRL::Normal(norm[0], norm[1], norm[2]));

        IRL::Paraboloid paraboloid = IRL::Paraboloid(datum,frame,y_pred[4].item<double>(),y_pred[5].item<double>());/*gen->new_parabaloid(y_pred[0].item<double>(),
        y_pred[1].item<double>(), y_pred[2].item<double>(), theta, phi, y_pred[3].item<double>(),
        y_pred[4].item<double>(), y_pred[5].item<double>());*/

        vector<IRL::Paraboloid> p;
        vector<IRL::Paraboloid> p1;
        double e = std::sqrt(DBL_EPSILON);

        datum = IRL::Pt(y_pred[0].item<double>()+e, y_pred[1].item<double>(), y_pred[2].item<double>());
        p.push_back(IRL::Paraboloid(datum,frame,y_pred[4].item<double>(),y_pred[5].item<double>())/*gen->new_parabaloid(y_pred[0].item<double>()+e, y_pred[1].item<double>(), y_pred[2].item<double>(), theta, phi, 
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>())*/);

        datum = IRL::Pt(y_pred[0].item<double>(), y_pred[1].item<double>()+e, y_pred[2].item<double>());
        p.push_back(IRL::Paraboloid(datum,frame,y_pred[4].item<double>(),y_pred[5].item<double>())/*gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>()+e, y_pred[2].item<double>(), theta, phi, 
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>())*/);

        datum = IRL::Pt(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>()+e);       
        p.push_back(IRL::Paraboloid(datum,frame,y_pred[4].item<double>(),y_pred[5].item<double>())/*gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>()+e, theta, phi, 
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>())*/);

        datum = IRL::Pt(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>()); 
        IRL::Pt temp2 = x_dir;
        IRL::Pt temp3 = x_dir;
        temp2[0] = cos(y_pred[3].item<double>()+e) * temp[0] + sin(y_pred[3].item<double>()) * y_dir[0];
        temp2[1] = cos(y_pred[3].item<double>()+e) * temp[1] + sin(y_pred[3].item<double>()) * y_dir[1];
        temp2[2] = cos(y_pred[3].item<double>()+e) * temp[2] + sin(y_pred[3].item<double>()) * y_dir[2];

        temp3[0] = norm[1] * temp2[2] - norm[2] * temp2[1];
        temp3[1] = -(norm[0] * temp2[2] - norm[2] * temp2[0]);
        temp3[2] = norm[0] * temp2[1] - norm[1] * temp2[0];
        IRL::ReferenceFrame frame2 = IRL::ReferenceFrame(IRL::Normal(temp2[0], temp2[1], temp2[2]), IRL::Normal(temp3[0], temp3[1], temp3[2]), IRL::Normal(norm[0], norm[1], norm[2]));
        p.push_back(IRL::Paraboloid(datum,frame2,y_pred[4].item<double>(),y_pred[5].item<double>())/*gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(), theta, phi, 
        y_pred[3].item<double>()+e, y_pred[4].item<double>(), y_pred[5].item<double>())*/);

        p.push_back(IRL::Paraboloid(datum,frame,y_pred[4].item<double>()+e,y_pred[5].item<double>())/*gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(), theta, phi, 
        y_pred[3].item<double>(), y_pred[4].item<double>()+e, y_pred[5].item<double>())*/);

        p.push_back(IRL::Paraboloid(datum,frame,y_pred[4].item<double>(),y_pred[5].item<double>()+e)/*gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(), theta, phi, 
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>()+e)*/);

        auto result = gen->get_fractions(paraboloid, false);

        vector<torch::Tensor> ep;
        vector<torch::Tensor> grads;

        for (int i = 0; i < 6; ++i)
        {
            ep.push_back(gen->get_fractions(p[i], false));
            torch::Tensor temp = torch::zeros(size);
            for (int j = 0; j < size; ++j)
            {
                temp[j] = (ep[i][j].item<double>() - result[j].item<double>()) / e;
            }
            grads.push_back(temp);
        }
        if (compute_requires_grad(y_pred)) 
        {
            auto grad_fn = std::shared_ptr<VolumeFracsNormalBackward>(new grad_functions::VolumeFracsNormalBackward(), deleteNode);

            grad_fn->set_next_edges(collect_next_edges(y_pred));
            grad_fn->frac_grads = grads;
            grad_fn->y_pred = y_pred;

            set_history(flatten_tensor_args(result), grad_fn);
        }
        return result;
    }
}

#endif