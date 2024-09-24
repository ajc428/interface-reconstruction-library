// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_H_

#include <torch/torch.h>
#include "mpi.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include "irl/machine_learning_reconstruction/neural_network.h"
#include "irl/machine_learning_reconstruction/grad_functions.h"
#include "irl/machine_learning_reconstruction/classification_neural_network.h"
#include "data_set.h"

using namespace std;

namespace IRL 
{
    std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
        {at::kByte, MPI_UNSIGNED_CHAR},
        {at::kChar, MPI_CHAR},
        {at::kDouble, MPI_DOUBLE},
        {at::kFloat, MPI_FLOAT},
        {at::kInt, MPI_INT},
        {at::kLong, MPI_LONG},
        {at::kShort, MPI_SHORT},
    };
    
    class trainer
    {
    private:
        int epochs = 100;
        int data_size = 100;
        int data_val_size = 100;
        double learning_rate = 0.001;
        int rank = 0;
        int numranks = 1;
        int batch_size = data_size / numranks;
        int val_batch_size = data_val_size / numranks;
        int type = 0;

        ofstream results_ex;
        ofstream results_pr;
        torch::Tensor train_in;
        torch::Tensor train_out;
        torch::Tensor val_in;
        torch::Tensor val_out;
        torch::Tensor test_in;
        torch::Tensor test_out;
        string train_in_file;
        string train_out_file;
        string test_in_file;
        string test_out_file;
        string validation_in_file;
        string validation_out_file;

        shared_ptr<IRL::model> nn;
        torch::nn::MSELoss critereon_MSE;
        torch::nn::BCELoss critereon_BCE;
        torch::nn::CrossEntropyLoss critereon_CE;
        torch::optim::Optimizer *optimizer;
        IRL::grad_functions *functions;
        
    public:
        trainer(int);
        trainer(int, int, double, int);
        ~trainer();
        void init();
        void load_train_data(string, string);
        void load_validation_data(string, string, int);
        void load_test_data(string, string);
        void load_model(string);
        void train_model(bool, string, string);
        void test_model(int);
        IRL::Normal get_normal(vector<double>*);
        IRL::Normal get_r2p_normal(vector<double>*);
        double get_type(vector<double>*);
        //vector<double> get_2normals(vector<double>*);
    };
}

#include "irl/machine_learning_reconstruction/trainer.tpp"

#endif