// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_CUDA_TPP_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_CUDA_TPP_

//#include "irl/interface_reconstruction_methods/elvira.h"
//#include "irl/interface_reconstruction_methods/reconstruction_interface.h"

namespace IRL
{
    trainer_cuda::trainer_cuda(int s)
    {             
        epochs = 0;
        data_size = 0;
        learning_rate = 0.001;
        m = s;
        
        if (m == 4)
        {
            nn = make_shared<model>(108,3,6);
            nn->to(device);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            critereon_MSE->to(device);
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 6)
        {
            nn = make_shared<model>(108,1,6);
            nn->to(device);
            nnn = make_shared<model>(108,1,6);
            nnn->to(device);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            critereon_MSE->to(device);
            functions = new IRL::grad_functions(6, m);
        }
        else
        {
            nn = make_shared<model>(108,8,2);
            nn->to(device);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            critereon_MSE->to(device);
            functions = new IRL::grad_functions(1, m);
        }
    }

    trainer_cuda::trainer_cuda(int e, int d, double l, int s)
    {           
        epochs = e;
        data_size = d;
        learning_rate = l;
        m = s;
        if (m == 4)
        {
            nn = make_shared<model>(108,3,6);
            nn->to(device);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            critereon_MSE->to(device);
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 6)
        {
            nn = make_shared<model>(108,1,6);
            nn->to(device);
            nnn = make_shared<model>(108,1,6);
            nnn->to(device);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            critereon_MSE->to(device);
            functions = new IRL::grad_functions(6, m);
        }
        else
        {
            nn = make_shared<model>(108,8,2);
            nn->to(device);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            critereon_MSE->to(device);
            functions = new IRL::grad_functions(1, m);
        }
    }

    trainer_cuda::~trainer_cuda()
    {
        delete optimizer;
        delete functions;
    }

    void trainer_cuda::load_train_data(string in_file, string out_file)
    {
        train_in_file = in_file;
        train_out_file = out_file;
    }

    void trainer_cuda::load_test_data(string in_file, string out_file)
    {
        test_in_file = in_file;
        test_out_file = out_file;
    }

    void trainer_cuda::train_model(bool load, std::string in, std::string out)
    {
        auto data_train = MyDataset(train_in_file, train_out_file, data_size, m);

        if (torch::cuda::is_available())
        {
            std::cout << "CUDA Activated" << std::endl;
        }
        
        if (load)
        {
            torch::load(nn, in);
        }
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double epoch_loss = 0;
            int count = 0;

            train_in = data_train.get_data().to(device);
            train_out = data_train.get_target().to(device);
            torch::Tensor y_pred = torch::zeros({data_size, 8}, device);

            y_pred = nn->forward(train_in).to(device);

            torch::Tensor check = torch::zeros({data_size, 8}, device);
            torch::Tensor comp = torch::zeros({data_size, 8}, device);
            if (m == 1)
            {
                check = torch::zeros({data_size, nn->getSize()}, device);
                comp = torch::zeros({data_size, nn->getSize()}, device);
                for (int i = 0; i < data_size; ++i)
                {
                    check[i] = functions->VolumeFracsForward(y_pred[i]).to(device);
                }
                comp = train_in;
            }
            else if (m == 2)
            {
                check = torch::zeros({data_size, nn->getSize()}, device);
                comp = torch::zeros({data_size, nn->getSize()}, device);
                for (int i = 0; i < data_size; ++i)
                {
                    check[i] = functions->VolumeFracsForwardFD(y_pred[i]).to(device);
                }
                comp = train_in;
            }
            else
            {
                check = torch::zeros({data_size, nn->getOutput()}, device);
                comp = torch::zeros({data_size, nn->getOutput()}, device);
                check = y_pred;
                comp = train_out;
            }

            torch::Tensor loss = torch::zeros({data_size, 1}, device);
           
            loss = critereon_MSE(check, comp);

            optimizer->zero_grad();
            loss.backward();

            optimizer->step();
            
            cout << epoch << " " << loss.item().toDouble() << endl;
                
            std::cout.flush();
            
        }

        torch::save(nn, out);
    }

    void trainer_cuda::test_model(int n)
    {
        auto data_test = MyDataset(test_in_file, test_out_file, data_size, 4/*m*/);
        results_ex.open("result_ex.txt");
        results_pr.open("result_pr.txt");

        if (n == 0)
        {
            nn->eval();
            for(int i = 0; i < data_test.size().value(); ++i)
            {
                test_in = data_test.get(i).data.to(device);
                test_out = data_test.get(i).target.to(device);
                torch::Tensor prediction = torch::zeros({8, 1}, device);
                prediction = nn->forward(test_in).to(device);
                
                for (int j = 0; j < 8; ++j)
                {
                    results_pr << prediction[j].item<double>() << " ";
                }
                for (int j = 0; j < 8; ++j)
                {
                    results_ex << test_out[j].item<double>() << " ";
                }
                results_ex << "\n";
                results_pr << "\n";
            }
        }
        else if (n == 1)
        {
            nn->eval();
            for(int i = 0; i < data_test.size().value(); ++i)
            {
                test_in = data_test.get(i).data.to(device);
                test_out = data_test.get(i).target.to(device);
                auto prediction = nn->forward(test_in).to(device);
                auto pred = functions->VolumeFracsForward(prediction).to(device);
                for (int j = 0; j < nn->getSize(); ++j)
                {
                    results_pr << pred[j].item<double>() << " ";
                }
                for (int j = 0; j < nn->getSize(); ++j)
                {
                    results_ex << test_in[j].item<double>() << " ";
                }
                results_ex << "\n";
                results_pr << "\n";
            }
        }
        else if (n == 2)
        {
            nn->eval();
            for(int i = 0; i < data_test.size().value(); ++i)
            {
                test_in = data_test.get(i).data.to(device);
                test_out = data_test.get(i).target.to(device);
                auto prediction = nn->forward(test_in).to(device);
                auto pred = functions->VolumeFracsForwardFD(prediction).to(device);
                for (int j = 0; j < nn->getSize(); ++j)
                {
                    results_pr << pred[j].item<double>() << " ";
                }
                for (int j = 0; j < nn->getSize(); ++j)
                {
                    results_ex << test_in[j].item<double>() << " ";
                }
                results_ex << "\n";
                results_pr << "\n";
            }
        }
        else if (n == 4)
        {
            nn->eval();
            loss_out.open("loss.txt");
            for(int i = 0; i < data_test.size().value(); ++i)
            {
                test_in = data_test.get(i).data.to(device);
                test_out = data_test.get(i).target.to(device);
                auto prediction = nn->forward(test_in).to(device);
                auto loss = critereon_MSE(prediction, test_out).to(device);
                for (int j = 0; j < 3; ++j)
                {
                    results_pr << prediction[j].item<double>() << " ";
                }
                for (int j = 0; j < 3; ++j)
                {
                    results_ex << test_out[j].item<double>() << " ";
                }
                loss_out << loss.item<double>() << "\n";
                results_ex << "\n";
                results_pr << "\n";
            }
            loss_out.close();
        }
        else if (n == 6)
        {
            nn->eval();
            for(int i = 0; i < data_test.size().value(); ++i)
            {
                test_in = data_test.get(i).data.to(device);
                test_out = data_test.get(i).target.to(device);
                auto prediction = nn->forward(test_in).to(device);
                results_pr << prediction.item<double>() << " ";
                results_ex << test_out.item<double>() << " ";
                results_ex << "\n";
                results_pr << "\n";
            }
            loss_out.close();
        }

        results_ex.close();
        results_pr.close();
    }


    void trainer_cuda::load_model(std::string in, int i)
    {
        if (i == 0)
        {
            torch::load(nn, in);
        }
        else if (i == 1)
        {
            torch::load(nnn, in);
        }
    }

    IRL::Normal trainer_cuda::get_normal(vector<double> fractions/*const DataMesh<double> liquid_volume_fraction, const DataMesh<IRL::Pt> liquid_centroid*/)
    {
        //vector<double> fractions;
        /*for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    fractions.push_back(liquid_volume_fraction(i, j, k));
                    fractions.push_back(liquid_centroid(i,j,k)[0]);
                    fractions.push_back(liquid_centroid(i,j,k)[1]);
                    fractions.push_back(liquid_centroid(i,j,k)[2]);
                }
            }
        }*/
        auto y_pred = nnn->forward(torch::tensor(fractions).to(device)).to(device);
        auto n = IRL::Normal();
        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        return n;
    }

    double trainer_cuda::get_normal_loss(vector<double> fractions/*const DataMesh<double> liquid_volume_fraction, const DataMesh<IRL::Pt> liquid_centroid*/)
    {
        /*vector<double> fractions;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    fractions.push_back(liquid_volume_fraction(i, j, k));
                    fractions.push_back(liquid_centroid(i,j,k)[0]);
                    fractions.push_back(liquid_centroid(i,j,k)[1]);
                    fractions.push_back(liquid_centroid(i,j,k)[2]);
                }
            }
        }*/

        auto y_pred = nnn->forward(torch::tensor(fractions).to(device)).to(device);
        return y_pred.item<double>();
    }
}

#endif