// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_SET_TPP_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_DATA_SET_TPP_

using namespace std;

vector<torch::Tensor> MyDataset::read_data(string file, int data_size, int m)
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

        if (m == 0)
        {
            output.push_back(torch::tensor(num));
        }
        else if (m == 3)
        {
            IRL::fractions *gen;
            IRL::spatial_moments *sm;
            gen = new IRL::fractions(3);
            sm = new IRL::spatial_moments();
            DataMesh<double> liquid_volume_fraction(gen->getMesh());
            DataMesh<IRL::Pt> liquid_centroid(gen->getMesh());
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        liquid_volume_fraction(i, j, k) = num[4*(i*9+j*3+k)];
                        liquid_centroid(i, j, k) = IRL::Pt(num[4*(i*9+j*3+k)+1], num[4*(i*9+j*3+k)+2], num[4*(i*9+j*3+k)+3]);
                    }
                }
            }
            output.push_back(sm->calculate_moments(liquid_volume_fraction, liquid_centroid, gen->getMesh()));
            delete gen;
            delete sm;
        }
        else if (/*m == 4 || m == 5*/false)
        {
            vector <double> num1;
            for (int i = 0; i < 27; ++i)
            {
                num1.push_back(num[i*4]);
            }
            output.push_back(torch::tensor(num1));
        }
        else
        {
            vector <double> num1;
            IRL::fractions *gen;
            IRL::spatial_moments *sm;
            gen = new IRL::fractions(3);
            sm = new IRL::spatial_moments();
            DataMesh<double> liquid_volume_fraction(gen->getMesh());
            DataMesh<IRL::Pt> liquid_centroid(gen->getMesh());
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        liquid_volume_fraction(i, j, k) = num[4*(i*9+j*3+k)];
                        liquid_centroid(i, j, k) = IRL::Pt(num[4*(i*9+j*3+k)+1], num[4*(i*9+j*3+k)+2], num[4*(i*9+j*3+k)+3]);
                    }
                }
            }
            sm->calculate_moments(liquid_volume_fraction, liquid_centroid, gen->getMesh());
            vector<double> centers = sm->get_mass_centers();
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        num1.push_back(liquid_volume_fraction(i, j, k));
                        if (liquid_volume_fraction(i, j, k) > 0 && liquid_volume_fraction(i, j, k) < 1)
                        {
                            num1.push_back(liquid_centroid(i, j, k)[0] - centers[0]);
                            num1.push_back(liquid_centroid(i, j, k)[1] - centers[1]);
                            num1.push_back(liquid_centroid(i, j, k)[2] - centers[2]);
                        }
                        else
                        {
                            num1.push_back(liquid_centroid(i, j, k)[0]);
                            num1.push_back(liquid_centroid(i, j, k)[1]);
                            num1.push_back(liquid_centroid(i, j, k)[2]);
                        }
                    }
                }
            }
            output.push_back(torch::tensor(num1));
            delete gen;
            delete sm;
        }
        ++i;
    }
    indata.close();
    return output;
}

torch::data::Example<> MyDataset::get(size_t index)
{
    return {data_in[index], data_out[index]};
} 

#endif