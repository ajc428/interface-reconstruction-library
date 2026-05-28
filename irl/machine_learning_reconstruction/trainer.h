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
#include "irl/machine_learning_reconstruction/data_set.h"

namespace IRL 
{
    inline std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
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

        std::ofstream results_ex;
        std::ofstream results_pr;
        std::ofstream results;
        torch::Tensor train_in;
        torch::Tensor train_out;
        torch::Tensor val_in;
        torch::Tensor val_out;
        torch::Tensor test_in;
        torch::Tensor test_out;
        std::string train_in_file;
        std::string train_out_file;
        std::string test_in_file;
        std::string test_out_file;
        std::string validation_in_file;
        std::string validation_out_file;

        std::shared_ptr<IRL::model> nn;
        torch::nn::MSELoss critereon_MSE;
        torch::nn::BCELoss critereon_BCE;
        torch::nn::CrossEntropyLoss critereon_CE;
        std::unique_ptr<torch::optim::Optimizer> optimizer;
        
    public:
        trainer(int);
        trainer(int, int, double, int);
        trainer(int, int, int, int, double, int);
        void init();
        void load_train_data(std::string, std::string);
        void load_validation_data(std::string, std::string, int);
        void load_test_data(std::string, std::string);
        void load_model(std::string);
        void train_model(bool, std::string, std::string);
        void test_model();
    };
}

#endif