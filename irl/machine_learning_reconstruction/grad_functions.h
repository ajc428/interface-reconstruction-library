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
        torch::Tensor MomentsForward(const torch::Tensor, DataMesh<double>&, DataMesh<IRL::Pt>&);
        torch::Tensor CurvatureForward(const torch::Tensor);
        torch::Tensor PLICForward(const torch::Tensor);
        
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

        struct MomentsBackward : public Node 
        {
            vector<torch::Tensor> moment_grads;
            torch::Tensor y_pred;

            variable_list apply(variable_list&& inputs) override 
            {
                Eigen::MatrixXd in_grads(12,1);
                Eigen::MatrixXd out_grads(8,12);
                Eigen::MatrixXd grad_result(8,1);
                for (int i = 0; i < 8; ++i)
                {
                    for (int j = 0; j < 12; ++j)
                    {
                        out_grads(i,j) = moment_grads[i][j].item<double>();
                    }
                }
                for (int i = 0; i < 12; ++i)
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

        struct CurvatureBackward : public Node 
        {
            vector<torch::Tensor> curvature_grads;
            torch::Tensor y_pred;

            variable_list apply(variable_list&& inputs) override 
            {
                Eigen::MatrixXd in_grads(1,1);
                Eigen::MatrixXd out_grads(2,1);
                Eigen::MatrixXd grad_result(2,1);
                for (int i = 0; i < 2; ++i)
                {
                    for (int j = 0; j < 1; ++j)
                    {
                        out_grads(i,j) = curvature_grads[i][j].item<double>();
                    }
                }
                for (int i = 0; i < 1; ++i)
                {
                    in_grads(i,0) = inputs[0][i].item<double>();
                }
                variable_list grad_inputs(1); 
                torch::Tensor temp = torch::zeros(2); 

                if (should_compute_output(0)) 
                {
                    grad_result = out_grads * in_grads;
                }

                for (int i = 0; i < 2; ++i)
                {
                    torch::Tensor temp2 = torch::zeros(1); 
                    temp2 = torch::tensor(grad_result(i,0));
                    temp[i] = temp2;
                }
                grad_inputs[0] = temp;
                return grad_inputs;
            }
        };

        struct PLICBackward : public Node 
        {
            vector<torch::Tensor> PLIC_grads;
            torch::Tensor y_pred;

            variable_list apply(variable_list&& inputs) override 
            {
                Eigen::MatrixXd in_grads(108,1);
                Eigen::MatrixXd out_grads(4,108);
                Eigen::MatrixXd grad_result(4,1);
                for (int i = 0; i < 4; ++i)
                {
                    for (int j = 0; j < 108; ++j)
                    {
                        out_grads(i,j) = PLIC_grads[i][j].item<double>();
                    }
                }
                for (int i = 0; i < 108; ++i)
                {
                    in_grads(i,0) = inputs[0][i].item<double>();
                }
                variable_list grad_inputs(1); 
                torch::Tensor temp = torch::zeros(4); 

                if (should_compute_output(0)) 
                {
                    grad_result = out_grads * in_grads;
                }

                for (int i = 0; i < 4; ++i)
                {
                    torch::Tensor temp2 = torch::zeros(1); 
                    temp2 = torch::tensor(grad_result(i,0));
                    temp[i] = temp2;
                }
                grad_inputs[0] = temp;
                return grad_inputs;
            }
        };
    };
}

#include "irl/machine_learning_reconstruction/grad_functions.tpp"

#endif