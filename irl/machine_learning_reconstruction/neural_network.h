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
    struct poolFirstMoments : torch::nn::Module 
    {
        int kernal;
        poolFirstMoments(int k)
        {
            kernal = k;
        }

        torch::Tensor forward(torch::Tensor x) 
        {
            int batch_size = x.size(0);
            int num_channels = x.size(1);
            int in_depth = x.size(2);
            int in_height = x.size(3);
            int in_width = x.size(4);
            int out_depth = in_depth-kernal+1;
            int out_height = in_height-kernal+1;
            int out_width = in_width-kernal+1;

            torch::Tensor y = torch::zeros({batch_size,num_channels,out_depth,out_height,out_width});

            //x=x.reshape({batch_size * num_channels, 1, in_depth, in_height, in_width});
            torch::Tensor regions = x.unfold(2, kernal, 1).unfold(3, kernal, 1).unfold(4, kernal, 1);
            regions = regions.reshape({-1, kernal*kernal*kernal});
            auto out = std::get<0>(regions.max(1));
            y = out.reshape({batch_size, num_channels, out_depth, out_height, out_width});
            // for (int i = 0; i < y.size(2); ++i)
            // {
            //     for (int j = 0; j < y.size(3); ++j)
            //     {
            //         for (int k = 0; k < y.size(4); ++k)
            //         {
            //             torch::Tensor region = x;  
            //             for (int n = 0; n < x.size(0); ++n)
            //             {
            //                 y[n][0][i][j][k] = torch::sum(x[n][0].narrow(0,i,kernal).narrow(1,j,kernal).narrow(2,k,kernal));
            //             }
            //         }
            //     }
            // }
            // y=y/(y.size(2)*y.size(3)*y.size(4));
            return y;
        }
    };

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

    struct model_cnn : torch::nn::Module 
    {
        int size;
        int out;
        int depth;
        int width;
        int type;
        model_cnn(int o, int d, int w) 
        {
            out = o;
            depth = d;
            width = w;
            //c1 = register_module("c1", torch::nn::Conv3d(torch::nn::Conv3dOptions(1, 1, 3).padding(1)));
            //c2 = register_module("c2", torch::nn::Conv3d(torch::nn::Conv3dOptions(1, 1, 3).padding(1)));
            //p1 = register_module("p1", torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2).padding(0).stride(1)));
            p2 = register_module("p2", torch::nn::AvgPool3d(torch::nn::AvgPool3dOptions(2).padding(0).stride(1)));
            p3 = register_module("p3", std::make_shared<poolFirstMoments>(2));
            l1 = register_module("l1", torch::nn::Linear(8, width));
            for (int i = 0; i < d-1; ++i)
            {
                layers.push_back(register_module("l" + std::to_string(i+2), torch::nn::Linear(width, width)));
            }
            lo = register_module("l" + std::to_string(d+1), torch::nn::Linear(width, out));
        }
        torch::Tensor forward(torch::Tensor x) 
        {
            // x = p2(torch::nn::functional::relu(c1(x)));
            // x = p2(torch::nn::functional::relu(c2(x)));
            // torch::Tensor d = torch::zeros({x.size(0),1,2,2,2});
            // for (int i = 0; i < x.size(0); ++i)
            // {
            //     double s = max(x[i][0]).item<double>();
            //     for (int ii = 0; ii < 2; ++ii)
            //     {
            //         for (int jj = 0; jj < 2; ++jj)
            //         {
            //             for (int kk = 0; kk < 2; ++kk)
            //             {
            //                 d[i][0][ii][jj][kk] = s;
            //             }
            //             //std::cout << s << std::endl;
            //         }
            //     }
            // }
            //x = torch::div(p2(x),d);
            x = p3->forward(x);
            x = torch::flatten(x, 1);
            x = torch::nn::functional::relu(l1(x));
            for (int i = 0; i < depth-1; ++i)
            {
                x = torch::nn::functional::relu(layers[i](x));
            }
            x = torch::sigmoid(lo(x));
            return x;
        }
        std::vector<torch::nn::Linear> layers;
        torch::nn::Linear l1{nullptr}, lo{nullptr};
        torch::nn::Conv3d c1{nullptr}, c2{nullptr};
        torch::nn::MaxPool3d p1{nullptr};
        torch::nn::AvgPool3d p2{nullptr};
        std::shared_ptr<IRL::poolFirstMoments> p3;

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