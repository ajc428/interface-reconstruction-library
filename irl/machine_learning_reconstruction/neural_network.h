// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <torch/torch.h>

namespace IRL 
{
    struct model : torch::nn::Module 
    {
        int size;
        int out;
        int depth;
        int width;
        int type;
        model(int s, int o, int d, int w, int t) 
        {
            size = s;
            out = o;
            depth = d;
            width = w;
            type = t;
            l1 = register_module("l1", torch::nn::Linear(size, width));
            for (int i = 0; i < d-1; ++i)
            {
                layers.push_back(register_module("l" + std::to_string(i+2), torch::nn::Linear(width, width)));
            }
            lo = register_module("l" + std::to_string(d+1), torch::nn::Linear(width, out));
        }
        torch::Tensor forward(torch::Tensor x) 
        {
            x = torch::nn::functional::relu(l1(x));
            for (int i = 0; i < depth-1; ++i)
            {
                x = torch::nn::functional::relu(layers[i](x));
            }
            switch (type)
            {
                case 1:
                    x = torch::sigmoid(lo(x));
                break;
                case 2:
                    x = torch::nn::functional::softmax(lo(x),-1);
                break;
                case 3:
                    x = torch::nn::functional::normalize(torch::square(lo(x)),torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
                break;
                default:
                    x = lo(x);
                break;
            }
            return x;
        }
        std::vector<torch::nn::Linear> layers;
        torch::nn::Linear l1{nullptr}, lo{nullptr};

        int getSize()
        {
            return size;
        }

        int getOutput()
        {
            return out;
        }
    };
}