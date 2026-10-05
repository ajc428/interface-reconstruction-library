// // This file is part of the Interface Reconstruction Library (IRL),
// // a library for interface reconstruction and computational geometry operations.
// //
// // Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
// //
// // This Source Code Form is subject to the terms of the Mozilla Public
// // License, v. 2.0. If a copy of the MPL was not distributed with this
// // file, You can obtain one at https://mozilla.org/MPL/2.0/.

// #ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_H_
// #define IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_H_

// #include <torch/torch.h>
// #include <torch/script.h>
// #include "mpi.h"
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <random>
// #include <string>
// #include "irl/machine_learning_reconstruction/neural_network.h"
// #include "irl/machine_learning_reconstruction/data_set.h"
// #include "irl/machine_learning_reconstruction/plic_refine.h"  

// namespace IRL 
// {
//     inline std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
//         {at::kByte, MPI_UNSIGNED_CHAR},
//         {at::kChar, MPI_CHAR},
//         {at::kDouble, MPI_DOUBLE},
//         {at::kFloat, MPI_FLOAT},
//         {at::kInt, MPI_INT},
//         {at::kLong, MPI_LONG},
//         {at::kShort, MPI_SHORT},
//     };
    
//     class trainer
//     {
//     private:
//         int epochs = 100;
//         int data_size = 100;
//         int data_val_size = 100;
//         double learning_rate = 0.001;
//         int rank = 0;
//         int numranks = 1;
//         int batch_size = data_size / numranks;
//         int val_batch_size = data_val_size / numranks;
//         int type = 0;

//         torch::jit::script::Module jit_nn;
//         bool use_jit = false;
//         bool vf_weighted_inputs = false;
//         std::string model_path;

//         std::ofstream results_ex;
//         std::ofstream results_pr;
//         std::ofstream results;
//         torch::Tensor train_in;
//         torch::Tensor train_out;
//         torch::Tensor val_in;
//         torch::Tensor val_out;
//         torch::Tensor test_in;
//         torch::Tensor test_out;
//         std::string train_in_file;
//         std::string train_out_file;
//         std::string test_in_file;
//         std::string test_out_file;
//         std::string validation_in_file;
//         std::string validation_out_file;

//         std::shared_ptr<IRL::model> nn;
//         torch::nn::MSELoss critereon_MSE;
//         torch::nn::BCELoss critereon_BCE;
//         torch::nn::CrossEntropyLoss critereon_CE;
//         std::unique_ptr<torch::optim::Optimizer> optimizer;
        
//     public:
//         trainer(int);
//         trainer(int, int, double, int);
//         trainer(int, int, int, int, double, int);
//         void init();
//         void load_train_data(std::string, std::string);
//         void load_validation_data(std::string, std::string, int);
//         void load_test_data(std::string, std::string);
//         void load_model(std::string);
//         void train_model(bool, std::string, std::string);
//         void test_model(std::string, std::string);
//         IRL::Normal predict_normal(const torch::Tensor& stencil_189);
//     };
// }

// #endif

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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
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
        // Per-rank minibatch size, and a real hyperparameter rather than
        // data_size/numranks. Gradients are allreduced before every optimizer
        // step, so the global batch is batch_size*numranks; setting this to
        // the per-rank dataset size gives one optimizer step per epoch.
        int batch_size = 256;
        int val_batch_size = 1024;
        int type = 0;

        // LR schedule: linear warmup then cosine decay. Batch scaling is off
        // by default so an existing known-good learning_rate keeps its meaning.
        double base_learning_rate = 0.001;
        int warmup_steps = 200;
        bool scale_lr_with_batch = false;
        double min_lr_factor = 0.01;

        // Horizon the cosine is sized against. If <= 0 this falls back to
        // steps_per_epoch*epochs, which is wrong whenever epochs is set as a
        // generous upper bound and early stopping ends the run far sooner:
        // the run then only traverses a small fraction of the cosine and the
        // LR stays effectively constant. Set this to where convergence is
        // actually expected so the decay completes.
        long schedule_steps = 0;

        // Alternative to cosine: multiply the LR by plateau_factor whenever
        // validation has not improved for plateau_patience evaluations. Robust
        // when the convergence horizon is not known in advance.
        bool use_plateau_decay = false;
        double plateau_factor = 0.5;
        int plateau_patience = 8;
        double plateau_min_lr_factor = 0.01;
        double plateau_lr_scale = 1.0;    // running product of decay factors
        int evals_since_plateau_drop = 0;

        // Validation / early-stopping cadence in optimizer steps, so it does
        // not silently change when batch_size does.
        int val_every_steps = 200;
        int patience_evals = 40;

        long global_step = 0;

        // Flat buffer so all parameter gradients go in one MPI_Allreduce
        // rather than one collective per parameter tensor.
        torch::Tensor grad_flat_buffer;
        std::vector<torch::Tensor> grad_params;

        void build_grad_buffer();
        void allreduce_gradients();
        double lr_at_step(long step, long total_steps) const;

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
        void test_model(std::string, std::string);

        // Batching / schedule configuration. Call before train_model().
        void set_batch_size(int b) { batch_size = b; };
        void set_val_batch_size(int b) { val_batch_size = b; };
        void set_warmup_steps(int w) { warmup_steps = w; };
        void set_scale_lr_with_batch(bool s) { scale_lr_with_batch = s; };
        void set_min_lr_factor(double f) { min_lr_factor = f; };
        // Size the cosine against an explicit step horizon rather than
        // steps_per_epoch*epochs. Pass roughly where you expect to converge.
        void set_schedule_steps(long s) { schedule_steps = s; };
        void set_plateau_decay(bool on, double factor = 0.5, int patience = 8)
        {
            use_plateau_decay = on;
            plateau_factor = factor;
            plateau_patience = patience;
        };
        void set_val_every_steps(int s) { val_every_steps = s; };
        void set_patience_evals(int p) { patience_evals = p; };
        int get_batch_size() const { return batch_size; };
        long get_global_step() const { return global_step; };
    };
}
#endif