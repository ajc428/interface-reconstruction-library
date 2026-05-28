// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_SET_H_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_SET_H_

#include <torch/torch.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        std::vector<torch::Tensor> data_in;
        std::vector<torch::Tensor> data_out;

    public:
        explicit MyDataset(std::string in_file, std::string out_file, int data_size)
        {
            data_in = read_data(in_file, data_size);
            data_out = read_data(out_file, data_size);
        };

        std::vector<torch::Tensor> read_data(std::string, int);
        torch::data::Example<> get(size_t index) override;
        torch::Tensor get_data();
        torch::Tensor get_target();
        torch::optional<size_t> size() const override 
        {
            return data_out.size();
        };
};

#endif