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
    trainer::trainer(int e, int d, double l, int s)
    {
        rank = MPI::COMM_WORLD.Get_rank();
        numranks = MPI::COMM_WORLD.Get_size();                      
        epochs = e;
        data_size = d;
        batch_size = data_size / numranks;
        learning_rate = l;
        m = s;
        if (m == 3)
        {
            nn = make_shared<model>(5);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
        }
        else if (m == 4)
        {
            nn_binary = make_shared<binary_model>(5);
            optimizer = new torch::optim::Adam(nn_binary->parameters(), learning_rate);
        }
        else
        {
            nn = make_shared<model>(108);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
        }
        critereon = torch::nn::MSELoss();
        critereon2 = torch::nn::BCELoss();
        functions = new IRL::grad_functions(3, m);
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

    void trainer::train_model(bool load, std::string in, std::string out)
    {
        cout << "Hello from rank " << rank << endl;
        auto data_train = MyDataset(train_in_file, train_out_file, data_size, m).map(torch::data::transforms::Stack<>());
        batch_size = data_train.size().value() / numranks;
        if (rank == 0)
        {
            cout << data_size << " " << batch_size << endl;
        }
        auto data_sampler = torch::data::samplers::DistributedRandomSampler(data_train.size().value(), numranks, rank, false);
        auto data_loader_train = torch::data::make_data_loader(std::move(data_train), data_sampler, batch_size);

        if (load)
        {
            if (m == 4)
            {
                torch::load(nn_binary, in);
            }
            else
            {
                torch::load(nn, in);
            }
        }
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double epoch_loss = 0;
            int count = 0;
            for (auto& batch : *data_loader_train)
            {
                train_in = batch.data;
                train_out = batch.target;

                torch::Tensor y_pred = torch::zeros({batch_size, 8});
                if (m == 4)
                {
                    y_pred = nn_binary->forward(train_in);
                }
                else
                {
                    y_pred = nn->forward(train_in);
                }
                torch::Tensor check = torch::zeros({batch_size, 8});
                torch::Tensor comp = torch::zeros({batch_size, 8});
                if (m == 0)
                {
                    check = y_pred;
                    comp = train_out;
                }
                else if (m == 1)
                {
                    check = torch::zeros({batch_size, nn->getSize()});
                    comp = torch::zeros({batch_size, nn->getSize()});
                    for (int i = 0; i < batch_size; ++i)
                    {
                        check[i] = functions->VolumeFracsForward(y_pred[i]);
                    }
                    comp = train_in;
                }
                else if (m == 2)
                {
                    check = torch::zeros({batch_size, nn->getSize()});
                    comp = torch::zeros({batch_size, nn->getSize()});
                    for (int i = 0; i < batch_size; ++i)
                    {
                        check[i] = functions->VolumeFracsForwardFD(y_pred[i]);
                    }
                    comp = train_in;
                }
                else if (m == 3)
                {
                    check = torch::zeros({batch_size, nn->getSize()});
                    comp = torch::zeros({batch_size, nn->getSize()});
                    for (int i = 0; i < batch_size; ++i)
                    {
                        IRL::fractions gen(3);
                        DataMesh<double> liquid_volume_fraction(gen.getMesh());
                        check[i] = functions->MomentsForward(y_pred[i], liquid_volume_fraction);
                    }
                    comp = train_in;
                }
                else if (m == 4)
                {
                    check = torch::zeros({batch_size, 1});
                    comp = torch::zeros({batch_size, 1});
                    check = y_pred;
                    comp = train_out;
                }
                
                auto loss = critereon(check, comp);
                if (m == 4)
                {
                    loss = critereon2(check, comp);
                    count = 0;
                    for (int i = 0; i < batch_size; ++i)
                    {
                        int x;
                        if (y_pred[i].item<double>() >= 0.5)
                        {
                            x = 1;
                        }
                        else
                        {
                            x = 0;
                        }
                        if (x == comp[i].item<int>())
                        {
                            ++count;
                        }
                    }
                }
                else
                {
                    epoch_loss = epoch_loss + loss.item().toDouble();
                }
                optimizer->zero_grad();
                loss.backward();
                
                if (numranks > 1)
                {
                    if (m == 4)
                    {
                        for (auto &param : nn_binary->named_parameters())
                        {
                            MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(), param.value().grad().numel(), mpiDatatype.at(param.value().grad().scalar_type()), MPI_SUM, MPI_COMM_WORLD);
                            param.value().grad().data() = param.value().grad().data()/numranks;
                        } 
                    }
                    else
                    {
                        for (auto &param : nn->named_parameters())
                        {
                            MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(), param.value().grad().numel(), mpiDatatype.at(param.value().grad().scalar_type()), MPI_SUM, MPI_COMM_WORLD);
                            param.value().grad().data() = param.value().grad().data()/numranks;
                        } 
                    }
                }

                optimizer->step();
            }
            if (rank == 0)
            {
                if (m == 4)
                {
                    cout << epoch << " " << count << "/" << batch_size << endl;
                }
                else
                {
                    cout << epoch << " " << epoch_loss << endl;
                }
                std::cout.flush();
            }
        }
        if (m == 4)
        {
            torch::save(nn_binary, out);
        }
        else
        {
            torch::save(nn, out);
        }

        MPI::Finalize(); 
    }

    void trainer::test_model(int n)
    {
        if (rank == 0)
        {
            auto data_test = MyDataset(test_in_file, test_out_file, data_size, m);
    
            results_ex.open("result_ex.txt");
            results_pr.open("result_pr.txt");
            if (n == 0)
            {
                nn->eval();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::zeros({8, 1});
                    prediction = nn->forward(test_in);
                    
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
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    auto pred = functions->VolumeFracsForward(prediction);
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
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    auto pred = functions->VolumeFracsForwardFD(prediction);
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
            else if (n == 3)
            {
                nn->eval();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    IRL::fractions gen(3);
                    DataMesh<double> liquid_volume_fraction(gen.getMesh());
                    auto pred = functions->MomentsForward(prediction, liquid_volume_fraction);
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
                nn_binary->eval();
                int count = 0;
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::zeros({1, 1});
                    prediction = nn_binary->forward(test_in);
                    
                    int x;
                    if (prediction[0].item<double>() >= 0.5)
                    {
                        x = 1;
                    }
                    else
                    {
                        x = 0;
                    }
                    if (x == test_out[0].item<int>())
                    {
                        ++count;
                    }

                    results_pr << prediction[0].item<double>() << " ";
                    results_ex << test_out[0].item<double>() << " ";

                    results_ex << "\n";
                    results_pr << "\n";
                }
                std::cout << "Result: " << count << "/" << data_test.size().value() << std::endl;
            }
            results_ex.close();
            results_pr.close();
        }
    }

    IRL::Paraboloid trainer::use_model(std::string in, const DataMesh<double> liquid_volume_fraction, const DataMesh<IRL::Pt> liquid_centroid)
    {
        vector<double> fractions;
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
        }

        IRL::fractions *gen = new IRL::fractions(3);;
        torch::load(nn, in);
        auto y_pred = nn->forward(torch::tensor(fractions));
        IRL::Paraboloid paraboloid = gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>());
        delete gen;
        return paraboloid;
    }
}

#endif