// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/machine_learning_reconstruction/data_set.h"

using namespace std;

vector<torch::Tensor> MyDataset::read_data(string file, int data_size)
{
    ifstream indata;
    vector<torch::Tensor> output;
    vector <double> num;
    indata.open(file);
    string line, value;
    int i = 0;
    while (getline(indata, line) && i < data_size)
    {
        num.clear();
        stringstream str(line);
        while (getline(str, value, ','))
        {
            num.push_back(stod(value));
        }

        output.push_back(torch::tensor(num));
        ++i;
    }
    indata.close();
    return output;
}

torch::data::Example<> MyDataset::get(size_t index)
{
    return {data_in[index], data_out[index]};
} 

torch::Tensor MyDataset::get_data()
{
    int len = data_in.size();
    int dim = data_in[0].size(0);
    torch::Tensor r = torch::zeros({len, dim});
    for (int i = 0; i < data_in.size(); ++i)
    {
        r[i] = data_in[i];
    }
    return r;
} 

torch::Tensor MyDataset::get_target()
{
    int len = data_out.size();
    int dim = data_out[0].size(0);
    torch::Tensor r = torch::zeros({len, dim});
    for (int i = 0; i < data_out.size(); ++i)
    {
        r[i] = data_out[i];
    }
    return r;
} 