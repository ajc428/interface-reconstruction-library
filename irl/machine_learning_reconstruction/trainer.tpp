// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_TPP_
#define IRL_MACHINE_LEARNING_RECONSTRUCTION_TRAINER_TPP_

namespace IRL
{
    trainer::trainer(int e, int d, double l)
    {
        rank = MPI::COMM_WORLD.Get_rank();
        numranks = MPI::COMM_WORLD.Get_size();                      
        epochs = e;
        data_size = d;
        batch_size = data_size / numranks;
        learning_rate = l;
        nn = make_shared<model>();
        critereon = torch::nn::MSELoss();
        optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
        functions = new IRL::grad_functions(3);
    }

    trainer::~trainer()
    {
        delete optimizer;
        delete functions;
    }

    void trainer::load_train_data(string in_file, string out_file)
    {
        train_in_file = in_file;
        train_out_file = out_file;
    }

    void trainer::load_test_data(string in_file, string out_file)
    {
        test_in_file = in_file;
        test_out_file = out_file;
    }

    void trainer::train_model(int m, bool load, std::string in, std::string out)
    {
        cout << "Hello from rank " << rank << endl;
        auto data_train = MyDataset(train_in_file, train_out_file, data_size).map(torch::data::transforms::Stack<>());
        batch_size = data_train.size().value() / numranks;
        if (rank == 0)
        {
            cout << data_size << " " << batch_size << endl;
        }
        auto data_sampler = torch::data::samplers::DistributedRandomSampler(data_train.size().value(), numranks, rank, false);
        auto data_loader_train = torch::data::make_data_loader(std::move(data_train), data_sampler, batch_size);

        if (load)
        {
            torch::load(nn, in);
        }
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double epoch_loss = 0;
            for (auto& batch : *data_loader_train)
            {
                train_in = batch.data;
                train_out = batch.target;

                auto y_pred = nn->forward(train_in);
                torch::Tensor check = torch::zeros({batch_size, 8});
                torch::Tensor comp = torch::zeros({batch_size, 8});
                if (m == 0)
                {
                    check = y_pred;
                    comp = train_out;
                }
                else if (m == 1)
                {
                    check = torch::zeros({batch_size, 108});
                    comp = torch::zeros({batch_size, 108});
                    for (int i = 0; i < batch_size; ++i)
                    {
                        check[i] = functions->VolumeFracsForward(y_pred[i]);
                    }
                    comp = train_in;
                }
                else if (m == 2)
                {
                    check = torch::zeros({batch_size, 108});
                    comp = torch::zeros({batch_size, 108});
                    for (int i = 0; i < batch_size; ++i)
                    {
                        check[i] = functions->VolumeFracsForwardFD(y_pred[i]);
                    }
                    comp = train_in;
                }
                auto loss = critereon(check, comp);
                epoch_loss = epoch_loss + loss.item().toDouble();

                optimizer->zero_grad();
                loss.backward();

                if (numranks > 1)
                {
                    for (auto &param : nn->named_parameters())
                    {
                        MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(), param.value().grad().numel(), mpiDatatype.at(param.value().grad().scalar_type()), MPI_SUM, MPI_COMM_WORLD);
                        param.value().grad().data() = param.value().grad().data()/numranks;
                    } 
                }

                optimizer->step();
            }
            if (rank == 0)
            {
                epoch_loss = epoch_loss;
                cout << epoch << " " << epoch_loss << endl;
                std::cout.flush();
            }
        }
        torch::save(nn, out);
        MPI::Finalize(); 
    }

    void trainer::test_model(int m)
    {
        if (rank == 0)
        {
            auto data_test = MyDataset(test_in_file, test_out_file, data_size);
            nn->eval();
    
            results.open("result.txt");
            if (m == 0)
            {
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    results << "Prediction\n";
                    for (int j = 0; j < 8; ++j)
                    {
                        results << prediction[j].item<double>() << " ";
                    }
                    results << "\nExpected\n";
                    for (int j = 0; j < 8; ++j)
                    {
                        results << test_out[j].item<double>() << " ";
                    }
                    results << "\n";
                }
            }
            else if (m == 1)
            {
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    auto pred = functions->VolumeFracsForward(prediction);
                    results << "Prediction\n";
                    for (int j = 0; j < 108; ++j)
                    {
                        results << pred[j].item<double>() << " ";
                    }
                    results << "\nExpected\n";
                    for (int j = 0; j < 108; ++j)
                    {
                        results << test_in[j].item<double>() << " ";
                    }
                    results << "\n";
                }
            }
            else if (m == 2)
            {
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    auto pred = functions->VolumeFracsForwardFD(prediction);
                    results << "Prediction\n";
                    for (int j = 0; j < 108; ++j)
                    {
                        results << pred[j].item<double>() << " ";
                    }
                    results << "\nExpected\n";
                    for (int j = 0; j < 108; ++j)
                    {
                        results << test_in[j].item<double>() << " ";
                    }
                    results << "\n";
                }
            }

            results.close();
        }
    }
}

#endif