// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/machine_learning_reconstruction/trainer.h"

namespace IRL
{
    trainer::trainer(int s)
    {
        rank = 0;
        numranks = 1;                      
        epochs = 0;
        data_size = 0;
        batch_size = 0;
        learning_rate = 0.001;
        type = s;
        init();
    }

    trainer::trainer(int e, int d, double l, int s)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numranks);                    
        epochs = e;
        data_size = d;
        batch_size = data_size / numranks;
        learning_rate = l;
        type = s;
        init();
    }

    trainer::trainer(int e, int d, int nh, int h, double l, int s)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numranks);                    
        epochs = e;
        data_size = d;
        batch_size = data_size / numranks;
        learning_rate = l;
        type = s;
        nn = std::make_shared<model>(189,3,nh,h,0);
        optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
        critereon_MSE = torch::nn::MSELoss();
        std::cout.precision(15);
    }

    void trainer::init()
    {
        std::cout.precision(15);
        switch (type)
        {
            case 0:
                nn = std::make_shared<model>(189,3,3,100,0);
                optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 1:
                nn = std::make_shared<model>(192,6,3,100,1);
                optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 2:
                nn = std::make_shared<model>(189,1,3,100,2);
                optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
                critereon_BCE = torch::nn::BCELoss();
            break;
            case 3:
                nn = std::make_shared<model>(189,3,5,100,3);
                optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
                //optimizer = new torch::optim::Adam(nn->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(0.001));
                critereon_MSE = torch::nn::MSELoss();
                critereon_BCE = torch::nn::BCELoss();
                critereon_CE = torch::nn::CrossEntropyLoss();
            break;
        }
    }

    void trainer::load_train_data(std::string in_file, std::string out_file)
    {
        train_in_file = in_file;
        train_out_file = out_file;
    }

    void trainer::load_validation_data(std::string in_file, std::string out_file, int x)
    {
        validation_in_file = in_file;
        validation_out_file = out_file;
        data_val_size = x;
    }

    void trainer::load_test_data(std::string in_file, std::string out_file)
    {
        test_in_file = in_file;
        test_out_file = out_file;
    }

    void trainer::train_model(bool load, std::string in, std::string out)
    {
        std::cout << "Hello from rank " << rank << std::endl;
        auto data_train = MyDataset(train_in_file, train_out_file, data_size).map(torch::data::transforms::Stack<>());
        batch_size = data_train.size().value() / numranks;
        if (rank == 0)
        {
            std::cout << data_size << " " << batch_size << std::endl;
        }
        auto data_sampler = torch::data::samplers::DistributedRandomSampler(data_train.size().value(), numranks, rank, false);
        auto data_loader_train = torch::data::make_data_loader(std::move(data_train), data_sampler, batch_size);

        auto data_val = MyDataset(validation_in_file, validation_out_file, data_val_size).map(torch::data::transforms::Stack<>());
        val_batch_size = data_val.size().value() / numranks;
        int val_size = data_val.size().value() / numranks;
        auto data_sampler_val = torch::data::samplers::DistributedRandomSampler(data_val.size().value(), numranks, rank, false);
        auto data_loader_val = torch::data::make_data_loader(std::move(data_val), data_sampler_val, val_size);

        if (load)
        {
            torch::load(nn, in);
        }
        double epoch_loss_val_check = 0;
        double lambda1 = 0.0;
        double lambda2 = 0.0;
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double epoch_loss = 0;
            double epoch_loss_val = 0;
            double total_epoch_loss = 0;
            double total_epoch_loss_val = 0;
            int count = 0;
            int size = 0;

            nn->train();
            size = nn->getOutput();

            for (auto& batch : *data_loader_train)
            {
                train_in = batch.data;
                train_out = batch.target;

                torch::Tensor y_pred = torch::zeros({batch_size, size});
                torch::Tensor check;
                torch::Tensor comp;

                y_pred = nn->forward(train_in);

                check = y_pred;
                comp = train_out;

                torch::Tensor loss = torch::zeros({batch_size, 1});
                torch::Tensor l2 = torch::tensor(0.0);
                torch::Tensor l1 = torch::tensor(0.0);
                for (auto &param : nn->named_parameters())
                {
                    if (param.key().find("weight") != std::string::npos)
                    {
                        l2 = l2 + lambda2 * param.value().square().sum();
                        l1 = l1 + lambda1 * param.value().abs().sum();
                    }
                }
                if (type == 1)
                {
                    auto loss_i = torch::nn::functional::mse_loss(check, comp, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone));
                    auto p1 = check.narrow(-1, 0, 3);
                    auto p2 = check.narrow(-1, 3, 3);
                    auto t1 = comp.narrow(-1, 0, 3);
                    auto t2 = comp.narrow(-1, 3, 3);
                    auto deadzone_v1 = (t1.norm(2, -1, true) < 0.1) & (p1.norm(2, -1, true) < 0.8);
                    auto deadzone_v2 = (t2.norm(2, -1, true) < 0.1) & (p2.norm(2, -1, true) < 0.8);
                    auto loss_v1 = loss_i.narrow(-1, 0, 3).masked_fill(deadzone_v1.expand_as(p1), 0.0);
                    auto loss_v2 = loss_i.narrow(-1, 3, 3).masked_fill(deadzone_v2.expand_as(p2), 0.0);
                    loss = torch::cat({loss_v1, loss_v2}, -1).mean();
                }
                else if (type == 2)
                {
                    loss = critereon_BCE(check, comp) + l2 + l1;
                    count += ((check > 0.5) == comp).sum().item<int>();
                }
                else if (type == 3)
                {
                    auto target_classes = torch::argmax(comp, 1);
                    loss = critereon_CE(check, target_classes) + l2 + l1;
                    count += torch::argmax(check, 1).eq(target_classes).sum().item<int>();
                }
                else
                {
                    loss = critereon_MSE(check, comp);
                }
                epoch_loss = epoch_loss + loss.item().toDouble()*batch.data.size(0);

                optimizer->zero_grad();
                loss.backward();

                if (numranks > 1)
                {
                    for (auto &param : nn->named_parameters())
                    {
                        MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(), param.value().grad().numel(), mpiDatatype.at(param.value().grad().scalar_type()), MPI_SUM, MPI_COMM_WORLD);
                        param.value().grad().div_(numranks);
                        
                    } 
                }

                optimizer->step();
            }

            if (numranks > 1)
            {
                MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }

            nn->eval();    
            
            for (auto& batch : *data_loader_val)
            {
                val_in = batch.data;
                val_out = batch.target;
                torch::Tensor y_pred = torch::zeros({val_batch_size, size});
                torch::Tensor check;
                torch::Tensor comp;

                y_pred = nn->forward(val_in);

                check = y_pred;
                comp = val_out;

                torch::Tensor loss = torch::zeros({val_batch_size, 1});
                torch::Tensor l2 = torch::tensor(0.0);
                torch::Tensor l1 = torch::tensor(0.0);
                for (auto &param : nn->named_parameters())
                {
                    if (param.key().find("weight") != std::string::npos)
                    {
                        l2 = l2 + lambda2 * param.value().square().sum();
                        l1 = l1 + lambda1 * param.value().abs().sum();
                    }
                }
                if (type == 1)
                {
                    auto loss_i = torch::nn::functional::mse_loss(check, comp, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone));
                    auto p1 = check.narrow(-1, 0, 3);
                    auto p2 = check.narrow(-1, 3, 3);
                    auto t1 = comp.narrow(-1, 0, 3);
                    auto t2 = comp.narrow(-1, 3, 3);
                    auto deadzone_v1 = (t1.norm(2, -1, true) < 0.1) & (p1.norm(2, -1, true) < 0.8);
                    auto deadzone_v2 = (t2.norm(2, -1, true) < 0.1) & (p2.norm(2, -1, true) < 0.8);
                    auto loss_v1 = loss_i.narrow(-1, 0, 3).masked_fill(deadzone_v1.expand_as(p1), 0.0);
                    auto loss_v2 = loss_i.narrow(-1, 3, 3).masked_fill(deadzone_v2.expand_as(p2), 0.0);
                    loss = torch::cat({loss_v1, loss_v2}, -1).mean();
                }
                else if (type == 2)
                {
                    loss = critereon_BCE(check, comp) + l2 + l1;
                }
                else if (type == 3)
                {
                    auto target_classes = torch::argmax(comp, 1);
                    loss = critereon_CE(check, target_classes) + l2 + l1;
                }
                else
                {
                    loss = critereon_MSE(check, comp);
                }
                epoch_loss_val = epoch_loss_val + loss.item().toDouble()*batch.data.size(0);
            }
            if (numranks == 1)
            {
                total_epoch_loss = epoch_loss;
                total_epoch_loss_val = epoch_loss_val;
            }
            else
            {
                MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&epoch_loss_val, &total_epoch_loss_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
            total_epoch_loss = total_epoch_loss / data_size;
            total_epoch_loss_val = total_epoch_loss_val / data_val_size;
            if (rank == 0)
            {
                if (type == 2 || type == 3)
                {
                    std::cout << epoch << " " << count << "/" << batch_size << std::endl;
                }
                std::cout << epoch << " " << total_epoch_loss << " " << total_epoch_loss_val << std::endl;
                std::cout.flush();
            }
            if (epoch % 100 == 0)
            {
                if (total_epoch_loss_val < epoch_loss_val_check || epoch == 0)
                {
                    epoch_loss_val_check = total_epoch_loss_val;
                }
                else if (epoch < epochs-1)
                {
                    epoch = epochs;
                }
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0 && (epoch % 100 == 0 || epoch == epochs - 1))
            {
                torch::save(nn, out);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } 
    }

    void trainer::test_model(std::string ex, std::string pr)
    {
        if (rank == 0)
        {
            auto data_test = MyDataset(test_in_file, test_out_file, data_size);
            results_ex.open(ex);
            results_pr.open(pr);
            int size = 0;

            if (type == 2)
            {
                nn->eval();
                size = 1;
                int count = 0;
                int total = data_test.size().value();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::zeros({1, 1});
                    prediction = nn->forward(test_in.unsqueeze(0));
                    if ((test_out[0].item<double>() == 1 && prediction[0].item<double>() > 0.5) || (test_out[0].item<double>() == 0 && prediction[0].item<double>() <= 0.5))
                    {
                        ++count;
                    }

                    results_pr << prediction[0].item<double>();
                    results_ex << test_out[0].item<double>();

                    results_ex << "\n";
                    results_pr << "\n";
                }
                std::cout << "Result: " << count << "/" << total << " (" << data_test.size().value() << ")" << std::endl;
            }
            else if (type == 3)
            {
                nn->eval();
                size = 3;
                int count = 0;
                int total = data_test.size().value();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::softmax(nn->forward(test_in.unsqueeze(0)), 1).squeeze(0);
                    auto ind = torch::argmax(test_out);
                    auto ind2 = torch::argmax(prediction);
                    if (ind.item<int>() == ind2.item<int>())
                    {
                        ++count;
                    }

                    for (int j = 0; j < size; ++j)
                    {
                        results_pr << prediction[j].item<double>() << " ";
                    }
                    for (int j = 0; j < size; ++j)
                    {
                        results_ex << test_out[j].item<double>() << " ";
                    }

                    results_ex << "\n";
                    results_pr << "\n";
                }
                std::cout << "Result: " << count << "/" << total << " (" << data_test.size().value() << ")" << std::endl;
            }
            else
            {
                nn->eval();
                size = nn->getOutput();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = nn->forward(test_in.unsqueeze(0)).squeeze(0);
                    
                    for (int j = 0; j < size; ++j)
                    {
                        results_pr << prediction[j].item<double>() << " ";
                    }
                    for (int j = 0; j < size; ++j)
                    {
                        results_ex << test_out[j].item<double>() << " ";
                    }
                    results_ex << "\n";
                    results_pr << "\n";
                }
            }
           
            results_ex.close();
            results_pr.close();
        }
    }

    void trainer::load_model(std::string in)
    {
        torch::load(nn,in);
    }
}