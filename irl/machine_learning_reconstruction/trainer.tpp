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

#include "irl/interface_reconstruction_methods/lvira_optimization.h"
#include "irl/interface_reconstruction_methods/lvira_neighborhood.h"
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
    
    trainer::~trainer()
    {
        delete optimizer;
        delete functions;
    }

    void trainer::init()
    {
        functions = new grad_functions(3,0);
        switch (type)
        {
            case 0:
                nn = make_shared<model>(189,8,3,100);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
            case 1:
                nn_binary = make_shared<binary_model>(189);
                optimizer = new torch::optim::Adam(nn_binary->parameters(), learning_rate);
                critereon_BCE = torch::nn::CrossEntropyLoss();
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 2:
                nn = make_shared<model>(189,3,3,100);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 3:
                nn = make_shared<model>(189,2,3,200);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                //torch::optim::AdamOptions(learning_rate).weight_decay(0.001)
                critereon_MSE = torch::nn::MSELoss();
            break;
        }
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
        auto data_train = MyDataset(train_in_file, train_out_file, data_size, type).map(torch::data::transforms::Stack<>());
        batch_size = data_train.size().value() / numranks;
        if (rank == 0)
        {
            cout << data_size << " " << batch_size << endl;
        }
        auto data_sampler = torch::data::samplers::DistributedRandomSampler(data_train.size().value(), numranks, rank, false);
        auto data_loader_train = torch::data::make_data_loader(std::move(data_train), data_sampler, batch_size);

        auto data_val = MyDataset(validation_in_file, validation_out_file, data_val_size, type).map(torch::data::transforms::Stack<>());
        val_batch_size = data_val.size().value() / numranks;
        double val_size = data_val.size().value() / numranks;
        auto data_sampler_val = torch::data::samplers::DistributedRandomSampler(data_val.size().value(), numranks, rank, false);
        auto data_loader_val = torch::data::make_data_loader(std::move(data_val), data_sampler_val, val_size);

        if (load)
        {
            if (type == 1)
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
            int size = 0;
            if (type == 1)
            {
                nn_binary->train();
                size = 1;
            }
            else
            {
                nn->train();
                size = nn->getOutput();
            }

            for (auto& batch : *data_loader_train)
            {
                train_in = batch.data;
                train_out = batch.target;

                torch::Tensor y_pred = torch::zeros({batch_size, size});
                torch::Tensor check;
                torch::Tensor comp;

                if (type == 1)
                {
                    y_pred = nn_binary->forward(train_in);
                }
                else
                {
                    y_pred = nn->forward(train_in);
                }

                if (type == 1000)
                {
                    check = functions->VolumeFracsForwardFD(y_pred);
                    comp = train_in;
                }
                else
                {
                    check = y_pred;
                    comp = train_out;
                }

                torch::Tensor loss = torch::zeros({batch_size, 1});
                if (type == 1)
                {
                    loss = critereon_MSE(check, comp);
                    count = 0;
                    for (int i = 0; i < batch_size; ++i)
                    {
                        if ((comp[i].item<double>() == 1 && check[i].item<double>() > 0.5) || (comp[i].item<double>() == 0 && check[i].item<double>() <= 0.5))
                        {
                            ++count;
                        }
                    }
                }
                else
                {
                    if (type == 3)
                    {
                        loss = functions->MSE_angle_loss(comp,check);
                    }
                    else
                    {
                        loss = critereon_MSE(check, comp);
                    }
                    epoch_loss = epoch_loss + loss.item().toDouble()*batch.data.size(0);
                }

                optimizer->zero_grad();
                loss.backward();

                if (numranks > 1)
                {
                    if (type == 1)
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

            if (type == 1)
            {
                nn_binary->eval();
            }
            else
            {
                nn->eval();
            }           
            
            for (auto& batch : *data_loader_val)
            {
                val_in = batch.data;
                val_out = batch.target;
                torch::Tensor y_pred = torch::zeros({val_batch_size, size});
                torch::Tensor check;
                torch::Tensor comp;

                if (type == 1)
                {
                    y_pred = nn_binary->forward(val_in);
                }
                else
                {
                    y_pred = nn->forward(val_in);
                }

                if (type == 1000)
                {
                    check = functions->VolumeFracsForwardFD(y_pred);
                    comp = val_in;
                }
                else
                {
                    check = y_pred;
                    comp = val_out;
                }

                torch::Tensor loss = torch::zeros({val_batch_size, 1});
                if (type == 1)
                {
                    loss = critereon_MSE(check, comp);
                    count = 0;
                    for (int i = 0; i < val_batch_size; ++i)
                    {
                        if ((comp[i].item<double>() == 1 && check[i].item<double>() > 0.5) || (comp[i].item<double>() == 0 && check[i].item<double>() <= 0.5))
                        {
                            ++count;
                        }
                    }
                }
                else
                {
                    if (type == 3)
                    {
                        loss = functions->MSE_angle_loss(comp,check);
                    }
                    else
                    {
                        loss = critereon_MSE(check, comp);
                    }
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
                if (type == 1)
                {
                    cout << epoch << " " << count << "/" << batch_size << endl;
                }
                cout << epoch << " " << total_epoch_loss << " " << total_epoch_loss_val << endl;
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
            if (rank == 0 && (epoch % 1000 == 0 || epoch == epochs - 1))
            {
                if (type == 1)
                {
                    torch::save(nn_binary, out);

                }
                else
                {
                    torch::save(nn, out);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Finalize(); 
    }

    void trainer::test_model(int n)
    {
        if (rank == 0)
        {
            auto data_test = MyDataset(test_in_file, test_out_file, data_size, type);
            results_ex.open("result_ex.txt");
            results_pr.open("result_pr.txt");
            int size = 0;

            if (n == 1)
            {
                nn_binary->eval();
                size = 1;
                int count = 0;
                int total = data_test.size().value();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::zeros({1, 1});
                    prediction = nn_binary->forward(test_in);
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
            else
            {
                nn->eval();
                size = nn->getOutput();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::zeros({size, 1});
                    prediction = nn->forward(test_in);
                    
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
            /*else if (n == 7)
            {
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;

                    IRL::LVIRANeighborhood<IRL::RectangularCuboid> neighborhood;
                    neighborhood.resize(27);
                    neighborhood.setCenterOfStencil(13);
                    IRL::RectangularCuboid cells[27];
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            for (int k = 0; k < 3; ++k) {
                                double* a = new double();
                                *a = test_in[7*(i*9+j*3+k)].item<double>();
                                const double* b = a;
                                    const int local_index =
                                        (k) * 9 + (j) * 3 + (i);
                                    cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
                                        IRL::Pt(i-1.5, j-1.5, k-1.5),
                                        IRL::Pt(i-0.5, j-0.5, k-0.5));
                                    neighborhood.setMember(
                                        static_cast<IRL::UnsignedIndex_t>(local_index),
                                        &cells[local_index], b);
                                    }
                                }
                            }
                            IRL::Pt a_gas_centroid = IRL::Pt(test_in[7*(1*9+1*3+1)+4].item<double>(), test_in[7*(1*9+1*3+1)+5].item<double>(), test_in[7*(1*9+1*3+1)+6].item<double>());
                            IRL::Pt a_liquid_centroid = IRL::Pt(test_in[7*(1*9+1*3+1)+1].item<double>(), test_in[7*(1*9+1*3+1)+2].item<double>(), test_in[7*(1*9+1*3+1)+3].item<double>());
                            auto bary_normal = IRL::Normal::fromPtNormalized(
                                a_gas_centroid - a_liquid_centroid);
                            bary_normal.normalize();
                            const double initial_distance =
                                bary_normal * neighborhood.getCenterCell().calculateCentroid();
                            IRL::PlanarSeparator a_interface = IRL::PlanarSeparator::fromOnePlane(
                                IRL::Plane(bary_normal, initial_distance));
                            IRL::setDistanceToMatchVolumeFractionPartialFill(
                                neighborhood.getCenterCell(),
                                neighborhood.getCenterCellStoredMoments(),
                                &a_interface);

                            a_interface =
                                IRL::reconstructionWithLVIRA3D(neighborhood, a_interface);
                    IRL::Normal n = a_interface[0].normal();
                    torch::Tensor prediction = torch::zeros(3);
                    prediction[0] = -n[0];
                    prediction[1] = -n[1];
                    prediction[2] = -n[2];

                    //auto prediction = nn->forward(test_in);
                    auto loss = critereon_MSE(prediction, test_out);
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
            else if (n == 8)
            {
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;

                    IRL::ELVIRANeighborhood neighborhood;
                    neighborhood.resize(27);
                    IRL::RectangularCuboid cells[27];
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            for (int k = 0; k < 3; ++k) {
                                double* a = new double();
                                *a = test_in[7*(i*9+j*3+k)].item<double>();
                                const double* b = a;
                                    const int local_index =
                                        (k) * 9 + (j) * 3 + (i);
                                    cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
                                        IRL::Pt(i-1.5, j-1.5, k-1.5),
                                        IRL::Pt(i-0.5, j-0.5, k-0.5));
                                    neighborhood.setMember(&cells[local_index], b,i-1,j-1,k-1);
                                    }
                                }
                            }

                            IRL::PlanarSeparator a_interface =
                                IRL::reconstructionWithELVIRA3D(neighborhood);
                    IRL::Normal n = a_interface[0].normal();
                    torch::Tensor prediction = torch::zeros(3);
                    prediction[0] = -n[0];
                    prediction[1] = -n[1];
                    prediction[2] = -n[2];

                    //auto prediction = nn->forward(test_in);
                    auto loss = critereon_MSE(prediction, test_out);
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
            }*/

            results_ex.close();
            results_pr.close();
        }
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

    IRL::Normal trainer::get_normal(vector<double>* fractions)
    {
        auto y_pred = nn->forward(torch::tensor(*fractions));
        auto n = IRL::Normal();
        n[0] = y_pred[0].item<double>();
        n[1] = y_pred[1].item<double>();
        n[2] = y_pred[2].item<double>();
        return n;
    }

    IRL::Normal trainer::get_r2p_normal(vector<double>* fractions)
    {
        auto y_pred = nn->forward(torch::tensor(*fractions));
        auto n = IRL::Normal();
        n[0] = cos(y_pred[1].item<double>()) * cos(y_pred[0].item<double>());
        n[1] = cos(y_pred[1].item<double>()) * sin(y_pred[0].item<double>());
        n[2] = sin(y_pred[1].item<double>());
        return n;
    }

    double trainer::get_type(vector<double>* fractions)
    {
        auto y_pred = nn_binary->forward(torch::tensor(*fractions));
        return y_pred[0].item<double>();
    }

    /*vector<double> trainer::get_2normals(vector<double>* fractions)
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
        }*
        auto y_pred = nn->forward(torch::tensor(*fractions));
        vector<double> normals;
        normals.push_back(y_pred[0].item<double>());
        normals.push_back(y_pred[1].item<double>());
        normals.push_back(y_pred[2].item<double>());
        normals.push_back(y_pred[3].item<double>());
        normals.push_back(y_pred[4].item<double>());
        normals.push_back(y_pred[5].item<double>());
        return normals;
    }*/
}

#endif