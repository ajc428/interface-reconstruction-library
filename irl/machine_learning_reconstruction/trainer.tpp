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

#include "irl/interface_reconstruction_methods/elvira.h"
#include "irl/interface_reconstruction_methods/reconstruction_interface.h"

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
            nn_binary = make_shared<binary_model>(3);
            optimizer = new torch::optim::Adam(nn_binary->parameters(), learning_rate);
            critereon_BCE = torch::nn::CrossEntropyLoss();
            functions = new IRL::grad_functions(3, m);
        }
        else if (m == 4)
        {
            nn = make_shared<model>(108,3,6);
            nnn = make_shared<model>(108,3,6);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(3, m);
        }
        else if (m == 5)
        {
            nn = make_shared<model>(108,6,2);
            nnn = make_shared<model>(108,3,6);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(3, m);
        }
        else
        {
            nn = make_shared<model>(108,8,2);
            nnn = make_shared<model>(108,3,6);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(3, m);
        }
    }

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
            nn_binary = make_shared<binary_model>(3);
            optimizer = new torch::optim::Adam(nn_binary->parameters(), learning_rate);
            critereon_BCE = torch::nn::CrossEntropyLoss();
            functions = new IRL::grad_functions(3, m);
        }
        else if (m == 4)
        {
            nn = make_shared<model>(108,3,6);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(3, m);
        }
        else if (m == 5)
        {
            nn = make_shared<model>(27,6,2);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(3, m);
        }
        else
        {
            nn = make_shared<model>(108,8,2);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(3, m);
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
            if (m == 3)
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
                    check = torch::zeros({batch_size, 3});
                    comp = torch::zeros({batch_size, 3});
                    check = y_pred;
                    comp = train_out;
                }
                else if (m == 5)
                {
                    check = torch::zeros({batch_size, nn->getSize()});
                    comp = torch::zeros({batch_size, nn->getSize()});
                    IRL::fractions *gen;
                    gen = new IRL::fractions(3);
                    DataMesh<double> liquid_volume_fraction(gen->getMesh());
                    DataMesh<IRL::Pt> liquid_centroid(gen->getMesh());
                    IRL::ELVIRANeighborhood neighborhood;
                    neighborhood.resize(27);
                    IRL::RectangularCuboid cells[27];
                    for (int n = 0; n < batch_size; ++n)
                    {
                        for (int i = 0; i < 3; ++i) 
                        {
                            for (int j = 0; j < 3; ++j) 
                            {
                                for (int k = 0; k < 3; ++k) 
                                {
                                    liquid_volume_fraction(i, j, k) = train_in[n][(9*i+3*j+k)].item<double>();
                                    //liquid_centroid(i, j, k) = IRL::Pt(train_in[n][4*(9*i+3*j+k)+1].item<double>(), train_in[n][4*(9*i+3*j+k)+2].item<double>(), train_in[n][4*(9*i+3*j+k)+3].item<double>());
                                    //comp[n][(9*i+3*j+k)] = train_in[n][4*(9*i+3*j+k)].item<double>();
                                    cells[k * 9 + j * 3 + i] = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(gen->getMesh().x(i), gen->getMesh().y(j), gen->getMesh().z(k)), IRL::Pt(gen->getMesh().x(i + 1), gen->getMesh().y(j + 1), gen->getMesh().z(k + 1)));
                                    neighborhood.setMember(&cells[(k) * 9 + (j) * 3 + (i)], &liquid_volume_fraction(i, j, k), i-1, j-1, k-1);
                                }
                            }
                        }
                        IRL::PlanarSeparator p = IRL::reconstructionWithELVIRA3D(neighborhood);
                        //IRL::Normal norm = this->get_normal("model_n.pt", liquid_volume_fraction, liquid_centroid);
                        //norm.normalize();
                        IRL::Normal norm = p[0].normal();
                        check[n] = functions->VolumeFracsNormalForward(y_pred[n], norm);
                    }
                    comp = train_in;
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
                        for (int j = 0; j < 3; ++j)
                        {
                            if (comp[i][j].item<double>() == 1)
                            {
                                for (int k = 0; k < 3; ++k)
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
                    epoch_loss = epoch_loss + loss.item().toDouble();
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
                    }
                }

                optimizer->step();
            }
            if (rank == 0)
            {
                if (m == 3)
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
        if (m == 3)
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
                invariants.open("invariants.txt");
                nn_binary->eval();
                int count = 0;
                int total = data_test.size().value();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    for (int j = 0; j < 3; ++j)
                    {
                        invariants << test_in[j].item<double>() << " ";
                    }
                    invariants << "\n";
                    torch::Tensor prediction = torch::zeros({1, 3});
                    prediction = nn_binary->forward(test_in);
                    
                    int x;
                    for (int j = 0; j < 3; ++j)
                    {
                        if (test_out[j].item<double>() == 1)
                        {
                            for (int k = 0; k < 3; ++k)
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

                    results_pr << prediction[0].item<double>() << " " << prediction[1].item<double>() << " " << prediction[2].item<double>();
                    results_ex << test_out[0].item<double>() << " " << test_out[1].item<double>() << " " << test_out[2].item<double>() << " ";

                    results_ex << "\n";
                    results_pr << "\n";
                }
                std::cout << "Result: " << count << "/" << total << " (" << data_test.size().value() << ")" << std::endl;
            }
            else if (n == 4)
            {
                nn->eval();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    auto prediction = nn->forward(test_in);
                    for (int j = 0; j < 3; ++j)
                    {
                        results_pr << prediction[j].item<double>() << " ";
                    }
                    for (int j = 0; j < 3; ++j)
                    {
                        results_ex << test_out[j].item<double>() << " ";
                    }
                    results_ex << "\n";
                    results_pr << "\n";
                }
            }
            else if (n == 5)
            {
                nn->eval();
                for(int h = 0; h < data_test.size().value(); ++h)
                {
                    test_in = data_test.get(h).data;
                    test_out = data_test.get(h).target;
                    auto prediction = nn->forward(test_in);

                    IRL::fractions *gen;
                    gen = new IRL::fractions(3);
                    IRL::ELVIRANeighborhood neighborhood;
                    neighborhood.resize(27);
                    IRL::RectangularCuboid cells[27];
                    DataMesh<double> liquid_volume_fraction(gen->getMesh());
                    DataMesh<IRL::Pt> liquid_centroid(gen->getMesh());
                    for (int i = 0; i < 3; ++i) 
                    {
                        for (int j = 0; j < 3; ++j) 
                        {
                            for (int k = 0; k < 3; ++k) 
                            {
                                liquid_volume_fraction(i, j, k) = test_in[(9*i+3*j+k)].item<double>();
                                cells[k * 9 + j * 3 + i] = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(gen->getMesh().x(i), gen->getMesh().y(j), gen->getMesh().z(k)), IRL::Pt(gen->getMesh().x(i + 1), gen->getMesh().y(j + 1), gen->getMesh().z(k + 1)));
                                neighborhood.setMember(&cells[(k) * 9 + (j) * 3 + (i)], &liquid_volume_fraction(i, j, k), i-1, j-1, k-1);
                            }
                        }
                    }
                    IRL::PlanarSeparator p = IRL::reconstructionWithELVIRA3D(neighborhood);
                    IRL::Normal norm = p[0].normal();
                    //IRL::Normal norm = get_normal("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model_n.pt", liquid_volume_fraction, liquid_centroid);
                    IRL::Pt x_dir = IRL::Pt(0,0,0);
                    if (abs(norm[0]) >= abs(norm[1]) && abs(norm[1]) >= abs(norm[2]))
                    {
                        x_dir[0] = norm[1];
                        x_dir[1] = -norm[0];
                        x_dir[2] = 0;
                    }
                    else if (abs(norm[1]) >= abs(norm[0]) && abs(norm[0]) >= abs(norm[2]))
                    {
                        x_dir[0] = -norm[1];
                        x_dir[1] = norm[0];
                        x_dir[2] = 0;
                    }
                    else if (abs(norm[0]) >= abs(norm[2]) && abs(norm[2]) >= abs(norm[1]))
                    {
                        x_dir[0] = norm[2];
                        x_dir[2] = -norm[0];
                        x_dir[1] = 0;
                    }
                    else if (abs(norm[1]) >= abs(norm[2]) && abs(norm[2]) >= abs(norm[0]))
                    {
                        x_dir[0] = 0;
                        x_dir[1] = norm[2];
                        x_dir[2] = -norm[1];
                    }
                    else if (abs(norm[2]) >= abs(norm[0]) && abs(norm[0]) >= abs(norm[1]))
                    {
                        x_dir[0] = -norm[2];
                        x_dir[2] = norm[0];
                        x_dir[1] = 0;
                    }
                    else if (abs(norm[2]) >= abs(norm[1]) && abs(norm[1]) >= abs(norm[0]))
                    {
                        x_dir[0] = 0;
                        x_dir[1] = -norm[2];
                        x_dir[2] = norm[1];
                    }
                    IRL::Pt y_dir = IRL::Pt(0,0,0);
                    y_dir[0] = norm[1] * x_dir[2] - norm[2] * x_dir[1];
                    y_dir[1] = -(norm[0] * x_dir[2] - norm[2] * x_dir[0]);
                    y_dir[2] = norm[0] * x_dir[1] - norm[1] * x_dir[0];

                    IRL::Pt temp = x_dir;
                    x_dir[0] = cos(prediction[3].item<double>()) * temp[0] + sin(prediction[3].item<double>()) * y_dir[0];
                    x_dir[1] = cos(prediction[3].item<double>()) * temp[1] + sin(prediction[3].item<double>()) * y_dir[1];
                    x_dir[2] = cos(prediction[3].item<double>()) * temp[2] + sin(prediction[3].item<double>()) * y_dir[2];

                    y_dir[0] = norm[1] * x_dir[2] - norm[2] * x_dir[1];
                    y_dir[1] = -(norm[0] * x_dir[2] - norm[2] * x_dir[0]);
                    y_dir[2] = norm[0] * x_dir[1] - norm[1] * x_dir[0];

                    IRL::ReferenceFrame frame = IRL::ReferenceFrame(IRL::Normal(x_dir[0], x_dir[1], x_dir[2]), IRL::Normal(y_dir[0], y_dir[1], y_dir[2]), IRL::Normal(norm[0], norm[1], norm[2]));

                    results_pr << frame[0] << " " << frame[1] << " " << frame[2] << " ";
                    for (int j = 0; j < 6; ++j)
                    {
                        if(j != 3)
                        {
                            results_pr << prediction[j].item<double>() << " ";
                        }
                    }
                    results_ex << test_out[6].item<double>() << " ";
                    results_ex << test_out[7].item<double>() << " ";
                    results_ex << "\n";
                    results_pr << "\n";
                }
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

    IRL::Paraboloid trainer::use_model2(std::string in, const DataMesh<double> liquid_volume_fraction, const DataMesh<IRL::Pt> liquid_centroid)
    {
        vector<double> fractions;
        IRL::fractions *gen;
        gen = new IRL::fractions(3);
        IRL::ELVIRANeighborhood neighborhood;
        neighborhood.resize(27);
        IRL::RectangularCuboid cells[27];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    fractions.push_back(liquid_volume_fraction(i, j, k));
                    cells[k * 9 + j * 3 + i] = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(gen->getMesh().x(i), gen->getMesh().y(j), gen->getMesh().z(k)), IRL::Pt(gen->getMesh().x(i + 1), gen->getMesh().y(j + 1), gen->getMesh().z(k + 1)));
                    neighborhood.setMember(&cells[(k) * 9 + (j) * 3 + (i)], &liquid_volume_fraction(i, j, k), i-1, j-1, k-1);
                }
            }
        }

        torch::load(nn, in);
        auto y_pred = nn->forward(torch::tensor(fractions));
        //IRL::Normal norm = get_normal("/home/andrew/Repositories/interface-reconstruction-library/examples/paraboloid_advector/model_n.pt", liquid_volume_fraction, liquid_centroid);
        IRL::PlanarSeparator p = IRL::reconstructionWithELVIRA3D(neighborhood);
        IRL::Normal norm = p[0].normal();
        IRL::Pt x_dir = IRL::Pt(0,0,0);
        if (abs(norm[0]) >= abs(norm[1]) && abs(norm[1]) >= abs(norm[2]))
        {
            x_dir[0] = norm[1];
            x_dir[1] = -norm[0];
            x_dir[2] = 0;
        }
        else if (abs(norm[1]) >= abs(norm[0]) && abs(norm[0]) >= abs(norm[2]))
        {
            x_dir[0] = -norm[1];
            x_dir[1] = norm[0];
            x_dir[2] = 0;
        }
        else if (abs(norm[0]) >= abs(norm[2]) && abs(norm[2]) >= abs(norm[1]))
        {
            x_dir[0] = norm[2];
            x_dir[2] = -norm[0];
            x_dir[1] = 0;
        }
        else if (abs(norm[1]) >= abs(norm[2]) && abs(norm[2]) >= abs(norm[0]))
        {
            x_dir[0] = 0;
            x_dir[1] = norm[2];
            x_dir[2] = -norm[1];
        }
        else if (abs(norm[2]) >= abs(norm[0]) && abs(norm[0]) >= abs(norm[1]))
        {
            x_dir[0] = -norm[2];
            x_dir[2] = norm[0];
            x_dir[1] = 0;
        }
        else if (abs(norm[2]) >= abs(norm[1]) && abs(norm[1]) >= abs(norm[0]))
        {
            x_dir[0] = 0;
            x_dir[1] = -norm[2];
            x_dir[2] = norm[1];
        }
        IRL::Pt y_dir = IRL::Pt(0,0,0);
        y_dir[0] = norm[1] * x_dir[2] - norm[2] * x_dir[1];
        y_dir[1] = -(norm[0] * x_dir[2] - norm[2] * x_dir[0]);
        y_dir[2] = norm[0] * x_dir[1] - norm[1] * x_dir[0];

        IRL::Pt temp = x_dir;
        x_dir[0] = cos(y_pred[3].item<double>()) * temp[0] + sin(y_pred[3].item<double>()) * y_dir[0];
        x_dir[1] = cos(y_pred[3].item<double>()) * temp[1] + sin(y_pred[3].item<double>()) * y_dir[1];
        x_dir[2] = cos(y_pred[3].item<double>()) * temp[2] + sin(y_pred[3].item<double>()) * y_dir[2];

        y_dir[0] = norm[1] * x_dir[2] - norm[2] * x_dir[1];
        y_dir[1] = -(norm[0] * x_dir[2] - norm[2] * x_dir[0]);
        y_dir[2] = norm[0] * x_dir[1] - norm[1] * x_dir[0];

        IRL::Pt datum = IRL::Pt(y_pred[0].item<double>(), y_pred[1].item<double>(), y_pred[2].item<double>());
        IRL::ReferenceFrame frame = IRL::ReferenceFrame(IRL::Normal(x_dir[0], x_dir[1], x_dir[2]), IRL::Normal(y_dir[0], y_dir[1], y_dir[2]), IRL::Normal(norm[0], norm[1], norm[2]));

        IRL::Paraboloid paraboloid = IRL::Paraboloid(datum,frame,y_pred[4].item<double>(),y_pred[5].item<double>());

        delete gen;
        return paraboloid;
    }

    void trainer::load_model(std::string in, int i)
    {
        if (i == 0)
        {
            torch::load(nn, in);
        }
        else
        {
            torch::load(nnn, in);
        }
    }

    IRL::Normal trainer::get_normal(const DataMesh<double> liquid_volume_fraction, const DataMesh<IRL::Pt> liquid_centroid)
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

        auto y_pred = nnn->forward(torch::tensor(fractions));
        auto n = IRL::Normal();
        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        return n;
    }

    IRL::ReferenceFrame trainer::getFrame(int num)
    {
        auto data_train = MyDataset(train_in_file, train_out_file, data_size, m);
        train_in = data_train.get(num).data;

        IRL::fractions gen(3);
        DataMesh<double> liquid_volume_fraction(gen.getMesh());
        DataMesh<IRL::Pt> liquid_centroid(gen.getMesh());
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    liquid_volume_fraction(i, j, k) = train_in[4*(i*9+j*3+k)].item<double>();
                    liquid_centroid(i, j, k) = IRL::Pt(train_in[4*(i*9+j*3+k)+1].item<double>(), train_in[4*(i*9+j*3+k)+2].item<double>(), train_in[4*(i*9+j*3+k)+3].item<double>());
                }
            }
        }

        Eigen::MatrixXd inertia(3,3);
        IRL::spatial_moments *sm = new IRL::spatial_moments();

        sm->calculate_moments(liquid_volume_fraction, liquid_centroid, gen.getMesh());
        vector<double> m = sm->getMoments();

        inertia(0,0) = m[1] + m[2];
        inertia(1,0) = -m[3];
        inertia(2,0) = -m[4];
        inertia(0,1) = -m[3];
        inertia(1,1) = m[0] + m[2];
        inertia(2,1) = -m[5];
        inertia(0,2) = -m[4];
        inertia(1,2) = -m[5];
        inertia(2,2) = m[0] + m[1];

        Eigen::EigenSolver<Eigen::MatrixXd> es;
        es.compute(inertia, true);
        double l1 = es.eigenvalues()[0].real();
        double l2 = es.eigenvalues()[1].real();
        double l3 = es.eigenvalues()[2].real();
        double lam_1;
        double lam_2;
        double lam_3;
        vector<double> ev1;
        vector<double> ev2;
        vector<double> ev3;
        if (l1 > l2)
        {
            if (l2 > l3)
            {
                lam_1 = l1;
                lam_2 = l2;
                lam_3 = l3;
                ev1.push_back(es.eigenvectors().col(0)[0].real());
                ev1.push_back(es.eigenvectors().col(0)[1].real());
                ev1.push_back(es.eigenvectors().col(0)[2].real());

                ev2.push_back(es.eigenvectors().col(1)[0].real());
                ev2.push_back(es.eigenvectors().col(1)[1].real());
                ev2.push_back(es.eigenvectors().col(1)[2].real());

                ev3.push_back(es.eigenvectors().col(2)[0].real());
                ev3.push_back(es.eigenvectors().col(2)[1].real());
                ev3.push_back(es.eigenvectors().col(2)[2].real());
            }
            else if (l1 > l3)
            {
                lam_1 = l1;
                lam_2 = l3;
                lam_3 = l2;
                ev1.push_back(es.eigenvectors().col(0)[0].real());
                ev1.push_back(es.eigenvectors().col(0)[1].real());
                ev1.push_back(es.eigenvectors().col(0)[2].real());

                ev2.push_back(es.eigenvectors().col(2)[0].real());
                ev2.push_back(es.eigenvectors().col(2)[1].real());
                ev2.push_back(es.eigenvectors().col(2)[2].real());

                ev3.push_back(es.eigenvectors().col(1)[0].real());
                ev3.push_back(es.eigenvectors().col(1)[1].real());
                ev3.push_back(es.eigenvectors().col(1)[2].real());
            }
            else
            {
                lam_1 = l3;
                lam_2 = l1;
                lam_3 = l2;
                ev1.push_back(es.eigenvectors().col(2)[0].real());
                ev1.push_back(es.eigenvectors().col(2)[1].real());
                ev1.push_back(es.eigenvectors().col(2)[2].real());

                ev2.push_back(es.eigenvectors().col(0)[0].real());
                ev2.push_back(es.eigenvectors().col(0)[1].real());
                ev2.push_back(es.eigenvectors().col(0)[2].real());

                ev3.push_back(es.eigenvectors().col(1)[0].real());
                ev3.push_back(es.eigenvectors().col(1)[1].real());
                ev3.push_back(es.eigenvectors().col(1)[2].real());
            }
        }
        else if (l3 > l2)
        {
            lam_1 = l3;
            lam_2 = l2;
            lam_3 = l1;
            ev1.push_back(es.eigenvectors().col(2)[0].real());
            ev1.push_back(es.eigenvectors().col(2)[1].real());
            ev1.push_back(es.eigenvectors().col(2)[2].real());

            ev2.push_back(es.eigenvectors().col(1)[0].real());
            ev2.push_back(es.eigenvectors().col(1)[1].real());
            ev2.push_back(es.eigenvectors().col(1)[2].real());

            ev3.push_back(es.eigenvectors().col(0)[0].real());
            ev3.push_back(es.eigenvectors().col(0)[1].real());
            ev3.push_back(es.eigenvectors().col(0)[2].real());
        }
        else if (l3 > l1)
        {
            lam_1 = l2;
            lam_2 = l3;
            lam_3 = l1;
            ev1.push_back(es.eigenvectors().col(1)[0].real());
            ev1.push_back(es.eigenvectors().col(1)[1].real());
            ev1.push_back(es.eigenvectors().col(1)[2].real());

            ev2.push_back(es.eigenvectors().col(2)[0].real());
            ev2.push_back(es.eigenvectors().col(2)[1].real());
            ev2.push_back(es.eigenvectors().col(2)[2].real());

            ev3.push_back(es.eigenvectors().col(0)[0].real());
            ev3.push_back(es.eigenvectors().col(0)[1].real());
            ev3.push_back(es.eigenvectors().col(0)[2].real());
        }
        else
        {
            lam_1 = l2;
            lam_2 = l1;
            lam_3 = l3;
            ev1.push_back(es.eigenvectors().col(1)[0].real());
            ev1.push_back(es.eigenvectors().col(1)[1].real());
            ev1.push_back(es.eigenvectors().col(1)[2].real());

            ev2.push_back(es.eigenvectors().col(0)[0].real());
            ev2.push_back(es.eigenvectors().col(0)[1].real());
            ev2.push_back(es.eigenvectors().col(0)[2].real());

            ev3.push_back(es.eigenvectors().col(2)[0].real());
            ev3.push_back(es.eigenvectors().col(2)[1].real());
            ev3.push_back(es.eigenvectors().col(2)[2].real());
        }

        Eigen::MatrixXd R(3,3);
        R(0,0) = ev1[0];
        R(0,1) = ev1[1];
        R(0,2) = ev1[2];
        R(1,0) = ev2[0];
        R(1,1) = ev2[1];
        R(1,2) = ev2[2];
        R(2,0) = ev3[0];
        R(2,1) = ev3[1];
        R(2,2) = ev3[2];

        IRL::ReferenceFrame frame = IRL::ReferenceFrame(IRL::Normal(1.0, 0.0, 0.0), IRL::Normal(0.0, 1.0, 0.0), IRL::Normal(0.0, 0.0, 1.0));
        frame[0] = IRL::Pt(R(0,0),R(0,1),R(0,2));
        frame[1] = IRL::Pt(R(1,0),R(1,1),R(1,2));
        frame[2] = IRL::Pt(R(2,0),R(2,1),R(2,2));
        
        return frame;
    }
}

#endif