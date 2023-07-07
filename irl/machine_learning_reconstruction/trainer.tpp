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
            nn_binary = make_shared<binary_model>(12);
            optimizer = new torch::optim::Adam(nn_binary->parameters(), learning_rate);
            critereon_BCE = torch::nn::BCELoss();
            functions = new IRL::grad_functions(3, m);
        }
        else if (m == 4)
        {
            nn = make_shared<model>(108,2);
            optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            critereon_MSE = torch::nn::MSELoss();
            functions = new IRL::grad_functions(3, m);
        }
        else
        {
            nn = make_shared<model>(108,8);
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
                if (m == 4)
                {
                    y_pred = torch::zeros({batch_size, 1});
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
                    check = torch::zeros({batch_size, 1});
                    comp = torch::zeros({batch_size, 1});
                    check = y_pred;
                    comp = train_out;
                }
                else if (m == 4)
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
                    for (int j = 0; j < 12; ++j)
                    {
                        invariants << test_in[j].item<double>() << " ";
                    }
                    invariants << "\n";
                    torch::Tensor prediction = torch::zeros({1, 1});
                    prediction = nn_binary->forward(test_in);
                    
                    int x;
                    if (prediction[0].item<double>() >= 0.5)
                    {
                        x = 1;
                    }
                    else if (prediction[0].item<double>() < 0.5)
                    {
                        x = 0;
                    }
                    else
                    {
                        x = 3;
                        --total;
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
                    for (int j = 0; j < 2; ++j)
                    {
                        results_pr << prediction[j].item<double>() << " ";
                    }
                    for (int j = 0; j < 2; ++j)
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