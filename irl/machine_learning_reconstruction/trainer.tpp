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

//#include "irl/interface_reconstruction_methods/elvira.h"
//#include "irl/interface_reconstruction_methods/reconstruction_interface.h"

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
        m = s;
        if (m == 3)
        {
            nn_binary = make_shared<binary_model>(108);
            optimizer = new torch::optim::Adam(nn_binary->parameters(), learning_rate);
            critereon_BCE = torch::nn::CrossEntropyLoss();
            //functions = new IRL::grad_functions(3, m);
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 4)
        {
            nn = make_shared<model>(108,3,3);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 5)
        {
            nn = make_shared<model>(189,3,3);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 6)
        {
            nn = make_shared<model>(189,6,3);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(4, m);
        }
        else
        {
            nn = make_shared<model>(108,8,2);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(1, m);
        }
    }

    trainer::trainer(int e, int d, double l, int s)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numranks);                    
        epochs = e;
        data_size = d;
        batch_size = data_size / numranks;
        learning_rate = l;
        m = s;
        if (m == 3)
        {
            nn_binary = make_shared<binary_model>(108);
            optimizer = new torch::optim::Adam(nn_binary->parameters(), learning_rate);
            critereon_BCE = torch::nn::CrossEntropyLoss();
            //functions = new IRL::grad_functions(3, m);
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 4)
        {
            nn = make_shared<model>(108,3,3);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 5)
        {
            nn = make_shared<model>(189,3,3);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(4, m);
        }
        else if (m == 6)
        {
            nn = make_shared<model>(189,6,3);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(4, m);
        }
        else
        {
            nn = make_shared<model>(108,8,2);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(1, m);
        }
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

    void trainer::load_validation_data(string in_file, string out_file, int x)
    {
        validation_in_file = in_file;
        validation_out_file = out_file;
        data_val_size = x;
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

        auto data_val = MyDataset(validation_in_file, validation_out_file, data_val_size, m).map(torch::data::transforms::Stack<>());
        val_batch_size = data_val.size().value() / numranks;
        double val_size = data_val.size().value() / numranks;
        auto data_sampler_val = torch::data::samplers::DistributedRandomSampler(data_val.size().value(), numranks, rank, false);
        auto data_loader_val = torch::data::make_data_loader(std::move(data_val), data_sampler_val, val_size);

        if (load)
        {
            if (m == 3)
            {
                torch::load(nn_binary, in);
            }
            else
            {
                torch::load(nn, in);
            }
        }
        double epoch_loss_val_check = 0;
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double epoch_loss = 0;
            double epoch_loss_val = 0;
            double total_epoch_loss = 0;
            double total_epoch_loss_val = 0;
            int count = 0;
            nn->train();
            for (auto& batch : *data_loader_train)
            {
                train_in = batch.data;
                train_out = batch.target;

                torch::Tensor y_pred = torch::zeros({batch_size, 8});
                if (m == 3)
                {
                    y_pred = torch::zeros({batch_size, 3});
                    y_pred = nn_binary->forward(train_in);
                }
                else
                {
                    y_pred = nn->forward(train_in);
                }

                torch::Tensor check = torch::zeros({batch_size, 8});
                torch::Tensor comp = torch::zeros({batch_size, 8});
                if (m == 1)
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
                    check = torch::zeros({batch_size, 2});
                    comp = torch::zeros({batch_size, 2});
                    check = y_pred;
                    comp = train_out;
                }
                else
                {
                    check = torch::zeros({batch_size, nn->getOutput()});
                    comp = torch::zeros({batch_size, nn->getOutput()});
                    check = y_pred;
                    comp = train_out;
                }

                torch::Tensor loss = torch::zeros({batch_size, 1});
                if (m == 3)
                {
                    loss = critereon_BCE(check, comp);
                    count = 0;
                    for (int i = 0; i < batch_size; ++i)
                    {
                        int x;
                        for (int j = 0; j < 2; ++j)
                        {
                            if (comp[i][j].item<double>() == 1)
                            {
                                for (int k = 0; k < 2; ++k)
                                {
                                    x = 1;
                                    if (y_pred[i][j].item<double>() < y_pred[i][k].item<double>())
                                    {
                                        x = 0;
                                        break;
                                    }
                                }
                            }
                        }
                        if (x == 1)
                        {
                            ++count;
                        }
                    }
                }
                else
                {
                    loss = critereon_MSE(check, comp);
                    epoch_loss = epoch_loss + loss.item().toDouble()*batch.data.size(0);
                }

                optimizer->zero_grad();
                loss.backward();
                
                if (numranks > 1)
                {
                    if (m == 3)
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
                        MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    }
                }

                optimizer->step();
            }
            nn->eval();
            for (auto& batch : *data_loader_val)
            {
                val_in = batch.data;
                val_out = batch.target;
                torch::Tensor y_pred = torch::zeros({val_batch_size, 8});
                if (m == 3)
                {
                    y_pred = torch::zeros({val_batch_size, 3});
                    y_pred = nn_binary->forward(val_in);
                }
                else
                {
                    y_pred = nn->forward(val_in);
                }

                torch::Tensor check = torch::zeros({val_batch_size, 8});
                torch::Tensor comp = torch::zeros({val_batch_size, 8});
                if (m == 1)
                {
                    check = torch::zeros({val_batch_size, nn->getSize()});
                    comp = torch::zeros({val_batch_size, nn->getSize()});
                    for (int i = 0; i < val_batch_size; ++i)
                    {
                        check[i] = functions->VolumeFracsForward(y_pred[i]);
                    }
                    comp = val_in;
                }
                else if (m == 2)
                {
                    check = torch::zeros({val_batch_size, nn->getSize()});
                    comp = torch::zeros({val_batch_size, nn->getSize()});
                    for (int i = 0; i < val_batch_size; ++i)
                    {
                        check[i] = functions->VolumeFracsForwardFD(y_pred[i]);
                    }
                    comp = val_in;
                }
                else if (m == 3)
                {
                    check = torch::zeros({val_batch_size, 2});
                    comp = torch::zeros({val_batch_size, 2});
                    check = y_pred;
                    comp = val_out;
                }
                else
                {
                    check = torch::zeros({val_batch_size, nn->getOutput()});
                    comp = torch::zeros({val_batch_size, nn->getOutput()});
                    check = y_pred;
                    comp = val_out;
                }

                torch::Tensor loss = torch::zeros({val_batch_size, 1});
                if (m == 3)
                {
                    loss = critereon_BCE(check, comp);
                }
                else
                {
                    loss = critereon_MSE(check, comp);
                    epoch_loss_val = epoch_loss_val + loss.item().toDouble()*batch.data.size(0);
                }

                if (numranks > 1)
                {
                    MPI_Allreduce(&epoch_loss_val, &total_epoch_loss_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                }
            }
            if (numranks == 1)
            {
                total_epoch_loss = epoch_loss;
                total_epoch_loss_val = epoch_loss_val;
            }
            total_epoch_loss = total_epoch_loss / data_size;
            total_epoch_loss_val = total_epoch_loss_val / data_val_size;
            if (rank == 0)
            {
                if (m == 3)
                {
                    cout << epoch << " " << count << "/" << val_batch_size << endl;
                }
                else
                {
                    cout << epoch << " " << total_epoch_loss << " " << total_epoch_loss_val << endl;
                }
                std::cout.flush();
            }
            if (epoch % 1000 == 0)
            {
                if (total_epoch_loss_val < epoch_loss_val_check || epoch == 0)
                {
                    epoch_loss_val_check = total_epoch_loss_val;
                    MPI_Bcast(&epoch_loss_val_check, 1, MPI_INT, 0, MPI_COMM_WORLD);
                }
                else if (epoch < epochs-1)
                {
                    epoch = epochs;
                    MPI_Bcast(&epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
                }
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0)
        {
            if (m == 3)
            {
                torch::save(nn_binary, out);

            }
            else
            {
                torch::save(nn, out);
            }
        }
        MPI_Finalize(); 
    }

    void trainer::test_model(int n)
    {
        if (rank == 0)
        {
            auto data_test = MyDataset(test_in_file, test_out_file, data_size, 4/*m*/);
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
                //invariants.open("invariants.txt");
                nn_binary->eval();
                int count = 0;
                int total = data_test.size().value();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    /*for (int j = 0; j < 3; ++j)
                    {
                        invariants << test_in[j].item<double>() << " ";
                    }
                    invariants << "\n";*/
                    torch::Tensor prediction = torch::zeros({1, 3});
                    prediction = nn_binary->forward(test_in);
                    
                    int x;
                    for (int j = 0; j < 2; ++j)
                    {
                        if (test_out[j].item<double>() == 1)
                        {
                            for (int k = 0; k < 2; ++k)
                            {
                                x = 1;
                                if (prediction[j].item<double>() < prediction[k].item<double>())
                                {
                                    x = 0;
                                    break;
                                }
                            }
                        }
                    }
                    if (x == 1)
                    {
                        ++count;
                    }

                    results_pr << prediction[0].item<double>() << " " << prediction[1].item<double>();// << " " << prediction[2].item<double>();
                    results_ex << test_out[0].item<double>() << " " << test_out[1].item<double>();// << " " << test_out[2].item<double>() << " ";

                    results_ex << "\n";
                    results_pr << "\n";
                }
                std::cout << "Result: " << count << "/" << total << " (" << data_test.size().value() << ")" << std::endl;
            }
            else if (n == 4)
            {
                nn->eval();
                loss_out.open("loss.txt");
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    auto loss = critereon_MSE(prediction, test_out);
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
                loss_out.open("loss.txt");
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    auto loss = critereon_MSE(prediction, test_out);
                    for (int j = 0; j < 6; ++j)
                    {
                        results_pr << prediction[j].item<double>() << " ";
                    }
                    for (int j = 0; j < 6; ++j)
                    {
                        results_ex << test_out[j].item<double>() << " ";
                    }
                    loss_out << loss.item<double>() << "\n";
                    results_ex << "\n";
                    results_pr << "\n";
                }
                loss_out.close();
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

        IRL::fractions *gen = new IRL::fractions(3);
        torch::load(nn, in);
        auto y_pred = nn->forward(torch::tensor(fractions));
        IRL::Paraboloid paraboloid = gen->new_parabaloid(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>(),
        y_pred[3].item<double>(), y_pred[4].item<double>(), y_pred[5].item<double>(),
        y_pred[6].item<double>(), y_pred[7].item<double>());
        delete gen;
        return paraboloid;
    }

    void trainer::load_model(std::string in, int i)
    {
        if (i == 0)
        {
            torch::load(nn, in);
        }
        else if (i == 1)
        {
            torch::load(nn_binary, in);
        }
    }

    IRL::Normal trainer::get_normal(vector<double>* fractions/*const DataMesh<double> liquid_volume_fraction, const DataMesh<IRL::Pt> liquid_centroid*/)
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
        auto y_pred = nn->forward(torch::tensor(*fractions));
        auto n = IRL::Normal();
        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        return n;
    }

    vector<double> trainer::get_2normals(vector<double>* fractions)
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
        auto y_pred = nn->forward(torch::tensor(*fractions));
        vector<double> normals;
        normals.push_back(y_pred[0].item<double>());
        normals.push_back(y_pred[1].item<double>());
        normals.push_back(y_pred[2].item<double>());
        normals.push_back(y_pred[3].item<double>());
        normals.push_back(y_pred[4].item<double>());
        normals.push_back(y_pred[5].item<double>());
        return normals;
    }
}

#endif