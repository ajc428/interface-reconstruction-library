// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_CUDA_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_CUDA_H_

#include <torch/torch.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include "irl/machine_learning_reconstruction/trainer.h"
#include "data_set.h"

using namespace std;

namespace IRL 
{
    class trainer_cuda
    {
    private:
        int epochs = 100;
        int data_size = 100;
        double learning_rate = 0.001;
        int m = 0;

        ofstream results_ex;
        ofstream results_pr;
        ofstream invariants;
        ofstream loss_out;
        torch::Tensor train_in;
        torch::Tensor train_in_rot;
        torch::Tensor train_out;
        torch::Tensor test_in;
        torch::Tensor test_in_rot;
        torch::Tensor test_out;
        string train_in_file;
        string train_out_file;
        string test_in_file;
        string test_out_file;

        shared_ptr<IRL::model> nn;
        shared_ptr<IRL::model> nnn;
        torch::nn::MSELoss critereon_MSE;
        torch::optim::Optimizer *optimizer;
        IRL::grad_functions *functions;

        torch::Device device = torch::kCUDA;
        
    public:
        trainer_cuda(int);
        trainer_cuda(int, int, double, int);
        ~trainer_cuda();
        void load_train_data(string, string);
        void load_test_data(string, string);
        void load_model(string, int);
        void set_train_data(vector<vector<double>>, vector<vector<double>>);
        void set_test_data(vector<vector<double>>, vector<vector<double>>);
        void train_model(bool, string, string);
        void test_model(int);
        IRL::Normal get_normal(vector<double>/*const DataMesh<double>, const DataMesh<IRL::Pt>*/);
        double get_normal_loss(vector<double>/*const DataMesh<double>, const DataMesh<IRL::Pt>*/);
    };
}

#include "irl/machine_learning_reconstruction/trainer_cuda.tpp"

#endif