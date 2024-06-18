// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_GRAD_FUNCTIONS_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_GRAD_FUNCTIONS_H_

#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 
#include <math.h>

#include "irl/machine_learning_reconstruction/fractions.h"
#include "irl/machine_learning_reconstruction/spatial_moments.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"

using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;

using torch::autograd::variable_list;
using torch::autograd::Variable;
using torch::autograd::tensor_list;

using torch::autograd::compute_requires_grad;
using torch::autograd::collect_next_edges;
using torch::autograd::flatten_tensor_args;

namespace IRL 
{
    class grad_functions
    {
    private:
        IRL::fractions *gen;
        IRL::spatial_moments *sm;
        int size;
    public:
        grad_functions(int, int);
        ~grad_functions();
        torch::Tensor VolumeFracsForward(const torch::Tensor);
        torch::Tensor VolumeFracsForwardFD(const torch::Tensor);
        torch::Tensor R2PVolumeFracsForward(const torch::Tensor,double,double,double);
        torch::Tensor VolumeFracsNormalForward(const torch::Tensor, IRL::Normal);
        
        torch::Tensor MSE_angle_loss(torch::Tensor out, torch::Tensor target)
        {   
            //torch::Tensor loss = torch::mse_loss(out,target);
            //torch::Tensor loss = torch::mean(torch::pow(torch::min(torch::fmod(out-target,2*M_PI),2*M_PI-torch::fmod(out-target,2*M_PI)),2));
            //torch::Tensor loss = torch::mean(torch::pow(torch::sin(out) - torch::sin(target),2)+torch::pow(torch::cos(out) - torch::cos(target),2));

            torch::Tensor loss = (torch::mse_loss(torch::cos(out.index({torch::indexing::Slice(),1})) * torch::cos(out.index({torch::indexing::Slice(),0})),torch::cos(target.index({torch::indexing::Slice(),1})) * torch::cos(target.index({torch::indexing::Slice(),0}))) + torch::mse_loss(torch::cos(out.index({torch::indexing::Slice(),1})) * torch::sin(out.index({torch::indexing::Slice(),0})),torch::cos(target.index({torch::indexing::Slice(),1})) * torch::sin(target.index({torch::indexing::Slice(),0}))) + torch::mse_loss(torch::sin(out.index({torch::indexing::Slice(),1})),torch::sin(target.index({torch::indexing::Slice(),1}))))/3;
            //+ (torch::mse_loss(torch::cos(out.index({torch::indexing::Slice(),3})) * torch::cos(out.index({torch::indexing::Slice(),2})),torch::cos(target.index({torch::indexing::Slice(),3})) * torch::cos(target.index({torch::indexing::Slice(),2}))) + torch::mse_loss(torch::cos(out.index({torch::indexing::Slice(),3})) * torch::sin(out.index({torch::indexing::Slice(),2})),torch::cos(target.index({torch::indexing::Slice(),3})) * torch::sin(target.index({torch::indexing::Slice(),2}))) + torch::mse_loss(torch::sin(out.index({torch::indexing::Slice(),3})),torch::sin(target.index({torch::indexing::Slice(),3}))))/3)/2;

            return loss;
        };

        struct VolumeFracsBackward : public Node 
        {
            vector<torch::Tensor> frac_grads;
            torch::Tensor y_pred;

            variable_list apply(variable_list&& inputs) override 
            {
                Eigen::MatrixXd in_grads(108,1);
                Eigen::MatrixXd out_grads(8,108);
                Eigen::MatrixXd grad_result(8,1);
                for (int i = 0; i < 8; ++i)
                {
                    for (int j = 0; j < 108; ++j)
                    {
                        out_grads(i,j) = frac_grads[i][j].item<double>();
                    }
                }
                for (int i = 0; i < 108; ++i)
                {
                    in_grads(i,0) = inputs[0][i].item<double>();
                }
                variable_list grad_inputs(1); 
                torch::Tensor temp = torch::zeros(8); 

                if (should_compute_output(0)) 
                {
                    grad_result = out_grads * in_grads;
                }

                for (int i = 0; i < 8; ++i)
                {
                    torch::Tensor temp2 = torch::zeros(1); 
                    temp2 = torch::tensor(grad_result(i,0));
                    temp[i] = temp2;
                }
                grad_inputs[0] = temp;
                return grad_inputs;
            }
        };

        struct VolumeFracsNormalBackward : public Node 
        {
            vector<torch::Tensor> frac_grads;
            torch::Tensor y_pred;

            variable_list apply(variable_list&& inputs) override 
            {
                Eigen::MatrixXd in_grads(108,1);
                Eigen::MatrixXd out_grads(6,108);
                Eigen::MatrixXd grad_result(6,1);
                for (int i = 0; i < 6; ++i)
                {
                    for (int j = 0; j < 108; ++j)
                    {
                        out_grads(i,j) = frac_grads[i][j].item<double>();
                    }
                }
                for (int i = 0; i < 108; ++i)
                {
                    in_grads(i,0) = inputs[0][i].item<double>();
                }
                variable_list grad_inputs(1); 
                torch::Tensor temp = torch::zeros(6); 

                if (should_compute_output(0)) 
                {
                    grad_result = out_grads * in_grads;
                }

                for (int i = 0; i < 6; ++i)
                {
                    torch::Tensor temp2 = torch::zeros(1); 
                    temp2 = torch::tensor(grad_result(i,0));
                    temp[i] = temp2;
                }
                grad_inputs[0] = temp;
                return grad_inputs;
            }
        };

        struct R2PBackward : public Node 
        {
            vector<torch::Tensor> R2P_grads;
            torch::Tensor y_pred;

            variable_list apply(variable_list&& inputs) override 
            {
                torch::Tensor in_grads = torch::zeros({189,1});
                torch::Tensor out_grads = torch::zeros({6,189});
                torch::Tensor grad_result = torch::zeros({6,1});
                for (int i = 0; i < 6; ++i)
                {
                    out_grads.index_put_({i,torch::indexing::Slice()},R2P_grads[i]);
                }
                in_grads = inputs[0];
                variable_list grad_inputs(1); 
                torch::Tensor temp = torch::zeros(6); 

                if (should_compute_output(0)) 
                {
                    grad_result = torch::matmul(out_grads, in_grads);
                }

                for (int i = 0; i < 6; ++i)
                {
                    temp[i] = grad_result[i];
                }
                grad_inputs[0] = temp;
                return grad_inputs;
            }
        };
    };
}

#include "irl/machine_learning_reconstruction/grad_functions.tpp"

#endif