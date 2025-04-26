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
        functions = new grad_functions(3,0);
        nn = make_shared<model>(189,3,nh,h,0);
        optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
        critereon_MSE = torch::nn::MSELoss();
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
                nn = make_shared<model>(189,3,3,100,0);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 1:
                nn = make_shared<model>(189,3,3,100,0);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 2:
                nn = make_shared<model>(189,1,3,100,0);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                //torch::optim::AdamOptions(learning_rate).weight_decay(0.001)
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 3:
                nn = make_shared<model>(3,1,5,100,1);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                //optimizer = new torch::optim::Adam(nn->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(0.001));
                critereon_BCE = torch::nn::BCELoss();
                critereon_MSE = torch::nn::MSELoss();
            break;
            case 4:
                nn = make_shared<model>(189,3,5,100,2);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                //optimizer = new torch::optim::Adam(nn->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(0.001));
                critereon_BCE = torch::nn::BCELoss();
                critereon_MSE = torch::nn::MSELoss();
                critereon_CE = torch::nn::CrossEntropyLoss();
            break;
            case 5:
                nn = make_shared<model>(189,27,3,100,3);
                optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
                critereon_MSE = torch::nn::MSELoss();
            break;
            // case 6:
            //     nn = make_shared<model>(189,1,3,100,0);
            //     //nn_ppic = make_shared<model_PPIC>(189,8,3,100,0);
            //     optimizer = new torch::optim::Adam(nn->parameters(), learning_rate);
            //     //optimizer = new torch::optim::Adam(nn_ppic->parameters(), learning_rate);
            //     critereon_MSE = torch::nn::MSELoss();
            // break;
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

                if (type == 1000)
                {
                    check = functions->VolumeFracsForwardFD(y_pred);
                    comp = train_in;
                }
                // else if (type == 6)
                // {
                //     check = torch::zeros({batch_size, 19});
                //     for (int i = 0; i < batch_size; ++i)
                //     {
                //         check[i] = torch::flatten(functions->MOF_Forward(y_pred[i]));
                //         //check[i] = torch::flatten(functions->MOF_Forward2(y_preds[0][i],y_preds[1][i],y_preds[2][i]));
                //     }
                //     comp = train_out;
                // }
                else if (type == 5)
                {
                    check = torch::zeros({batch_size, 3});
                    comp = torch::zeros({batch_size, 3});
                    for (int i = 0; i < batch_size; ++i)
                    {
                        check[i] = torch::flatten(functions->LVIRAForward(y_pred[i], train_in[i]));
                    }
                    comp = train_out;
                }
                else
                {
                    check = y_pred;
                    comp = train_out;
                }

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
                if (type == 3)
                {
                    loss = critereon_BCE(check, comp) + l2 + l1;
                    count = 0;
                    for (int i = 0; i < batch_size; ++i)
                    {
                        if ((comp[i].item<double>() == 1 && check[i].item<double>() > 0.5) || (comp[i].item<double>() == 0 && check[i].item<double>() <= 0.5))
                        {
                            ++count;
                        }
                    }
                }
                else if (type == 4)
                {
                    loss = critereon_CE(check, comp) + l2 + l1;
                    count = 0;
                    for (int i = 0; i < batch_size; ++i)
                    {
                        auto ind = torch::argmax(comp[i]);
                        auto ind2 = torch::argmax(check[i]);

                        if (ind.item<int>() == ind2.item<int>())
                        {
                            ++count;
                        }
                    }
                }
                else if (type == 2)
                {
                    loss = functions->MSE_angle_loss(comp,check) + l2 + l1;
                }
                else
                {
                    loss = critereon_MSE(check, comp);
                    //loss = /*critereon_MSE(check.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)}), comp.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)}));// + */functions->MSE_paraboloid_ref(check,comp) + l2 + l1;
                }
                epoch_loss = epoch_loss + loss.item().toDouble()*batch.data.size(0);

                optimizer->zero_grad();
                loss.backward();

                if (numranks > 1)
                {
                    for (auto &param : nn->named_parameters())
                    {
                        MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(), param.value().grad().numel(), mpiDatatype.at(param.value().grad().scalar_type()), MPI_SUM, MPI_COMM_WORLD);
                        param.value().grad().data() = param.value().grad().data()/numranks;
                        
                    } 
                    MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                }

                optimizer->step();
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

                if (type == 1000)
                {
                    check = functions->VolumeFracsForwardFD(y_pred);
                    comp = val_in;
                }
                // else if (type == 6)
                // {
                //     check = torch::zeros({batch_size, 19});
                //     for (int i = 0; i < batch_size; ++i)
                //     {
                //         check[i] = torch::flatten(functions->MOF_Forward(y_pred[i]));
                //         //check[i] = torch::flatten(functions->MOF_Forward2(y_preds[0][i],y_preds[1][i],y_preds[2][i]));
                //     }
                //     comp = train_out;
                // }
                else if (type == 5)
                {
                    check = torch::zeros({val_batch_size, 3});
                    comp = torch::zeros({val_batch_size, 3});
                    for (int i = 0; i < val_batch_size; ++i)
                    {
                        check[i] = torch::flatten(functions->LVIRAForward(y_pred[i], val_in[i]));
                    }
                    comp = val_out;
                }
                else
                {
                    check = y_pred;
                    comp = val_out;
                }

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
                if (type == 3)
                {
                    loss = critereon_BCE(check, comp) + l2 + l1;
                }
                else if (type == 4)
                {
                    loss = critereon_CE(check, comp) + l2 + l1;
                }
                else if (type == 2)
                {
                    loss = functions->MSE_angle_loss(comp,check) + l2 + l1;
                }
                else
                {
                    loss = critereon_MSE(check, comp);
                    //loss = /*critereon_MSE(check.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)}), comp.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)}));// + */functions->MSE_paraboloid_ref(check,comp) + l2 + l1;
                }
                epoch_loss_val = epoch_loss_val + loss.item().toDouble()*batch.data.size(0);

                if (numranks > 1)
                {
                    MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
                if (type == 3 || type == 4)
                {
                    cout << epoch << " " << count << "/" << batch_size << endl;
                }
                cout << epoch << " " << total_epoch_loss << " " << total_epoch_loss_val << endl;
                std::cout.flush();
            }
            if (epoch % 100 == 0)
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
            if (rank == 0 && (epoch % 100 == 0 || epoch == epochs - 1))
            {
                torch::save(nn, out);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } 
    }

    void trainer::test_model(int n)
    {
        if (rank == 0)
        {
            auto data_test = MyDataset(test_in_file, test_out_file, data_size, type);
            results_ex.open("result_ex.txt");
            results_pr.open("result_pr.txt");
            //results.open("results.txt");
            int size = 0;

            if (n == 3)
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
                    prediction = nn->forward(test_in);
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
            else if (type == 4)
            {
                nn->eval();
                size = 3;
                int count = 0;
                int total = data_test.size().value();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::softmax(nn->forward(test_in),0);
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
            else if (type == 5)
            {
                nn->eval();
                size = nn->getOutput();
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor weights = torch::zeros({27, 1});
                    torch::Tensor prediction = torch::zeros({size, 1});
                    weights = nn->forward(test_in);
                    prediction = functions->LVIRAForward(weights, test_in);
                    
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
            // else if (type == 6)
            // {
            //     //nn_ppic->eval();
            //     nn->eval();
            //     size = nn->getOutput();
            //     for(int i = 0; i < data_test.size().value(); ++i)
            //     {
            //         test_in = data_test.get(i).data;
            //         test_out = data_test.get(i).target;
            //         torch::Tensor moments = torch::zeros({19, 1});
            //         torch::Tensor prediction = torch::zeros({2, 1});
            //         vector<torch::Tensor> predictions;

            //         //prediction = nn_ppic->forward(test_in);
            //         prediction = nn->forward(test_in);
            //         //moments = functions->MOF_Forward(prediction);
            //         //moments = functions->MOF_Forward2(predictions[0],predictions[1],predictions[2]);
                    
            //         for (int j = 0; j < 1; ++j)
            //         {
            //             results_pr << prediction[j].item<double>() << " ";
            //         }
            //         for (int j = 0; j < 1; ++j)
            //         {
            //             results_ex << test_out[j].item<double>() << " ";
            //         }
            //         for (int j = 0; j < 8; ++j)
            //         {
            //             //results << prediction[j].item<double>() << " ";
            //         }
            //         // for (int j = 0; j < 3; ++j)
            //         // {
            //         //     results << predictions[1][j].item<double>() << " ";
            //         // }
            //         // for (int j = 0; j < 2; ++j)
            //         // {
            //         //     results << predictions[2][j].item<double>() << " ";
            //         // }
            //         results_ex << "\n";
            //         results_pr << "\n";
            //         //results << "\n";
            //     }
            // }

            else if (n == 9)
            {
                double e = 0;
                double e20 = 0;
                for(int y = 0; y < data_test.size().value(); ++y)
                {
                    double e1 = 0;
                    test_in = data_test.get(y).data;
                    test_out = data_test.get(y).target;

                    std::vector<IRL::Polygon> poly;

                    for (int i = 1; i < 4; ++i) {
                        for (int j = 1; j < 4; ++j) {
                            for (int k = 1; k < 4; ++k) {
                    IRL::LVIRANeighborhood<IRL::RectangularCuboid> neighborhood;
                    neighborhood.resize(27);
                    neighborhood.setCenterOfStencil(13);
                    IRL::RectangularCuboid cells[27];
                    int local_index = 0;
                    for (int ii = i-1; ii < i+2; ++ii) {
                        for (int jj = j-1; jj < j+2; ++jj) {
                            for (int kk = k-1; kk < k+2; ++kk) {
                                double* a = new double();
                                *a = test_in[7*(ii*25+jj*5+kk)].item<double>();
                                const double* b = a;
                                    cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
                                        IRL::Pt(ii-2.5, jj-2.5, kk-2.5),
                                        IRL::Pt(ii-1.5, jj-1.5, kk-1.5));
                                    neighborhood.setMember(
                                        static_cast<IRL::UnsignedIndex_t>(local_index),
                                        &cells[local_index], b);
                                        local_index++;
                                    }
                                }
                            }
                            IRL::PlanarSeparator a_interface;
                            if (test_in[7*(i*25+j*5+k)].item<double>() < IRL::global_constants::VF_LOW || test_in[7*(i*25+j*5+k)].item<double>() > IRL::global_constants::VF_HIGH)
                            {
                                const double distance = std::copysign(1.0, test_in[7*(i*25+j*5+k)].item<double>() - 0.5);
                                a_interface = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
                            }
                            else
                            {
                                IRL::Pt a_gas_centroid = IRL::Pt(test_in[7*(i*25+j*5+k)+4].item<double>(), test_in[7*(i*25+j*5+k)+5].item<double>(), test_in[7*(i*25+j*5+k)+6].item<double>());
                                IRL::Pt a_liquid_centroid = IRL::Pt(test_in[7*(i*25+j*5+k)+1].item<double>(), test_in[7*(i*25+j*5+k)+2].item<double>(), test_in[7*(i*25+j*5+k)+3].item<double>());
                                auto bary_normal = IRL::Normal::fromPtNormalized(
                                    a_gas_centroid - a_liquid_centroid);
                                bary_normal.normalize();
                                const double initial_distance =
                                    bary_normal * neighborhood.getCenterCell().calculateCentroid();
                                a_interface = IRL::PlanarSeparator::fromOnePlane(
                                    IRL::Plane(bary_normal, initial_distance));
                                IRL::setDistanceToMatchVolumeFractionPartialFill(
                                    neighborhood.getCenterCell(),
                                    neighborhood.getCenterCellStoredMoments(),
                                    &a_interface);
                                a_interface =
                                    IRL::reconstructionWithLVIRA3D(neighborhood, a_interface);
                            }
                            poly.push_back(IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(neighborhood.getCenterCell(), a_interface, a_interface[0]));
                            }
                        }
                    }
                    IRL::Normal norm_poly = poly[13].calculateNormal();
                    //std::cout << norm_poly << std::endl;
                    IRL::ReferenceFrame fit_frame;
                    int largest_dir = 0;
                    if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[1]))
                    largest_dir = 1;
                    if (std::fabs(norm_poly[largest_dir]) < std::fabs(norm_poly[2]))
                    largest_dir = 2;
                    if (largest_dir == 0)
                    fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 1.0, 0.0));
                    else if (largest_dir == 1)
                    fit_frame[0] = crossProduct(norm_poly, IRL::Normal(0.0, 0.0, 1.0));
                    else
                    fit_frame[0] = crossProduct(norm_poly, IRL::Normal(1.0, 0.0, 0.0));
                    fit_frame[0].normalize();
                    fit_frame[1] = crossProduct(norm_poly, fit_frame[0]);
                    fit_frame[2] = norm_poly;
                    const IRL::Pt lower_cell_pt(-0.5, -0.5, -0.5);
                    const IRL::Pt upper_cell_pt(0.5, 0.5, 0.5);
                    const IRL::Pt cell_center = 0.5 * (lower_cell_pt + upper_cell_pt);
                    IRL::Paraboloid paraboloid;
                    double sum_vfrac = 0.0;
                    for (int i = 1; i < 4; ++i) {
                        for (int j = 1; j < 4; ++j) {
                            for (int k = 1; k < 4; ++k) {
                                double* a = new double();
                                sum_vfrac += test_in[7*(i*25+j*5+k)].item<double>();
                            }
                        }
                    }

                    const double meshsize = 1.0;  // mesh.dx();
                    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
                    Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
                    const IRL::Pt pref = poly[13].calculateCentroid();//IRL::Pt(test_in[7*(1*9+1*3+1)+1].item<double>(), test_in[7*(1*9+1*3+1)+2].item<double>(), test_in[7*(1*9+1*3+1)+3].item<double>());
                    const auto frame = fit_frame;
                    int step = 0;
                    for (int i = 1; i < 4; ++i) {
                      for (int j = 1; j < 4; ++j) {
                        for (int k = 1; k < 4; ++k) {
                          const IRL::UnsignedIndex_t shape =
                              poly[step].getNumberOfVertices();
                          if (shape == 0) {
                            step++;
                            continue;
                          }
                          // Local polygon normal and centroid
                          IRL::Pt ploc = poly[step].calculateCentroid();//IRL::Pt(test_in[7*(i*9+j*3+k)+1].item<double>(), test_in[7*(i*9+j*3+k)+2].item<double>(), test_in[7*(i*9+j*3+k)+3].item<double>());
                          IRL::Normal nloc = poly[step].calculateNormal();
                          // if (frame[2] * nloc <= 0.0) {
                          //   continue;
                          // }
                          ploc -= pref;
                          ploc /= meshsize;
                          const IRL::Pt tmp_pt = ploc;
                          const IRL::Normal tmp_n = nloc;
                          for (IRL::UnsignedIndex_t d = 0; d < 3; ++d) {
                            ploc[d] = frame[d] * tmp_pt;
                            nloc[d] = frame[d] * tmp_n;
                          }
                          // Plane coefficients
                          Eigen::VectorXd reconstruction_plane_coeffs(3);
                          reconstruction_plane_coeffs << -(ploc * nloc) / meshsize, nloc[0],
                              nloc[1];
                          reconstruction_plane_coeffs /= -nloc[2];
                          // Integrals
                          Eigen::VectorXd integrals = Eigen::VectorXd::Zero(6);
                          double b_dot_sum = 0.0;
                          for (IRL::UnsignedIndex_t v = 0; v < shape; ++v) {
                            IRL::UnsignedIndex_t vn = (v + 1) % shape;
                            IRL::Pt vert1 = poly[step][v];
                            IRL::Pt vert2 = poly[step][vn];
                            vert1 -= pref;
                            vert2 -= pref;
                            vert1 /= meshsize;
                            vert2 /= meshsize;
                            IRL::Pt tmp_pt1 = vert1;
                            IRL::Pt tmp_pt2 = vert2;
                            for (IRL::UnsignedIndex_t d = 0; d < 3; ++d) {
                              vert1[d] = frame[d] * tmp_pt1;
                              vert2[d] = frame[d] * tmp_pt2;
                            }

                            const double xv = vert1[0];
                            const double yv = vert1[1];
                            const double xvn = vert2[0];
                            const double yvn = vert2[1];

                            Eigen::VectorXd integral_to_add(6);
                            integral_to_add << (xv * yvn - xvn * yv) / 2.0,
                                (xv + xvn) * (xv * yvn - xvn * yv) / 6.0,
                                (yv + yvn) * (xv * yvn - xvn * yv) / 6.0,
                                (xv + xvn) * (xv * xv + xvn * xvn) * (yvn - yv) / 12.0,
                                (yvn - yv) *
                                    (3.0 * xv * xv * yv + xv * xv * yvn + 2.0 * xv * xvn * yv +
                                     2.0 * xv * xvn * yvn + xvn * xvn * yv +
                                     3.0 * xvn * xvn * yvn) /
                                    24.0,
                                (xv - xvn) * (yv + yvn) * (yv * yv + yvn * yvn) / 12.0;
                            integrals += integral_to_add;
                          }
                          b_dot_sum += integrals.head(3).dot(reconstruction_plane_coeffs);
                          // Get weighting
                          const double gaussianweight =  1.0;
                          const double vfrac = test_in[7*(i*25+j*5+k)].item<double>();
                          double vfrac_weight = 1.0;
                          const double limit_vfrac = 0.1;
                          if (vfrac < limit_vfrac) {
                            vfrac_weight = 0.5 - 0.5 * std::cos(M_PI * vfrac / limit_vfrac);
                          } else if (vfrac > 1.0 - limit_vfrac) {
                            vfrac_weight =
                                0.5 - 0.5 * std::cos(M_PI * (1.0 - vfrac) / limit_vfrac);
                          }
                          double ww = 1.0;
                          ww *= gaussianweight;
                          ww *= vfrac_weight;
                          if (ww > 0.0) {
                            A += ww * integrals * integrals.transpose();
                            b += ww * integrals * b_dot_sum;
                          }
                          step++;
                        }
                      }
                    }

                    Eigen::VectorXd sol = A.colPivHouseholderQr().solve(b);
                    std::array<double, 6> sol_fit = {
                        {sol(0), sol(1), sol(2), sol(3), sol(4), sol(5)}};

                    const double a1 = sol_fit[0], b1 = sol_fit[1], c1 = sol_fit[2],
                    d1 = sol_fit[3], e2 = sol_fit[4], f1 = sol_fit[5];
                    const double theta = 0.5 * std::atan2(e2, (IRL::safelyTiny(d1 - f1)));
                    const double cos_t = std::cos(theta);
                    const double sin_t = std::sin(theta);
                    const double A1 =
                    -(d1 * cos_t * cos_t + f1 * sin_t * sin_t + e2 * cos_t * sin_t);
                    const double B =
                    -(f1 * cos_t * cos_t + d1 * sin_t * sin_t - e2 * cos_t * sin_t);
                    torch::Tensor prediction = torch::zeros(3);
                    prediction[0] = d1;
                    prediction[1] = f1;
                    prediction[2] = e2;
                    
                    auto loss = critereon_MSE(prediction, test_out);
                    for (int j = 0; j < 3; ++j)
                    {
                        results_pr << prediction[j].item<double>() << " ";
                        e1 = e1 + pow((prediction[j].item<double>() - test_out[j].item<double>()),2.0);
                    }
                    for (int j = 0; j < 3; ++j)
                    {
                        results_ex << test_out[j].item<double>() << " ";
                    }
                    double theta10 = atan2(-prediction[0].item<double>()+prediction[1].item<double>()+sqrt(prediction[0].item<double>()*prediction[0].item<double>()-2*prediction[0].item<double>()*prediction[1].item<double>()+prediction[1].item<double>()*prediction[1].item<double>()+prediction[2].item<double>()*prediction[2].item<double>()),prediction[2].item<double>());
                    double AA = prediction[0].item<double>()*cos(theta10)*cos(theta10) + prediction[1].item<double>()*sin(theta10)*sin(theta10) + prediction[2].item<double>()*cos(theta10)*sin(theta10);
                    double BB = prediction[0].item<double>()*sin(theta10)*sin(theta10) + prediction[1].item<double>()*cos(theta10)*cos(theta10) - prediction[2].item<double>()*cos(theta10)*sin(theta10);
                    double theta20 = atan2(-test_out[0].item<double>()+test_out[1].item<double>()+sqrt(test_out[0].item<double>()*test_out[0].item<double>()-2*test_out[0].item<double>()*test_out[1].item<double>()+test_out[1].item<double>()*test_out[1].item<double>()+test_out[2].item<double>()*test_out[2].item<double>()),test_out[2].item<double>());
                    double AA1 = test_out[0].item<double>()*cos(theta20)*cos(theta20) + test_out[1].item<double>()*sin(theta20)*sin(theta20) + test_out[2].item<double>()*cos(theta20)*sin(theta20);
                    double BB1 = test_out[0].item<double>()*sin(theta20)*sin(theta20) + test_out[1].item<double>()*cos(theta20)*cos(theta20) - test_out[2].item<double>()*cos(theta20)*sin(theta20);
                    e20 = e20 + pow((4*AA*BB - 4*AA1*BB1),2.0)/pow(4*AA1*BB1,2.0);
                    //e20 = e20 + pow((4*prediction[0].item<double>()*prediction[1].item<double>() - 4*test_out[0].item<double>()*test_out[1].item<double>()),2.0)/pow(4*test_out[0].item<double>()*test_out[1].item<double>(),2.0);
                    //e20 = e20 + pow(prediction[0].item<double>()- test_out[0].item<double>(),2.0) + pow(prediction[1].item<double>() - test_out[1].item<double>(),2.0);
                    //e20 = e20/2;
                    e1 = e1 / 3;
                    e = e + e1;
                    results_ex << "\n";
                    results_pr << "\n";
                }
                e = e / data_test.size().value();
                e20 = sqrt(e20 / data_test.size().value());
                std::cout << e20 << ","<< std::flush;
            }

            else
            {
                nn->eval();
                size = nn->getOutput();
                double e = 0;
                double e2 = 0;
                for(int i = 0; i < data_test.size().value(); ++i)
                {
                    double e1 = 0;
                    test_in = data_test.get(i).data;
                    test_out = data_test.get(i).target;
                    torch::Tensor prediction = torch::zeros({size, 1});
                    prediction = nn->forward(test_in);
                    
                    for (int j = 0; j < size; ++j)
                    {
                        results_pr << prediction[j].item<double>() << " ";
                        e1 = e1 + pow((prediction[j].item<double>() - test_out[j].item<double>()),2.0);
                    }
                    for (int j = 0; j < size; ++j)
                    {
                        results_ex << test_out[j].item<double>() << " ";
                    }
                    double theta = atan2(-prediction[0].item<double>()+prediction[1].item<double>()+sqrt(prediction[0].item<double>()*prediction[0].item<double>()-2*prediction[0].item<double>()*prediction[1].item<double>()+prediction[1].item<double>()*prediction[1].item<double>()+prediction[2].item<double>()*prediction[2].item<double>()),prediction[2].item<double>());
                    double AA = prediction[0].item<double>()*cos(theta)*cos(theta) + prediction[1].item<double>()*sin(theta)*sin(theta) + prediction[2].item<double>()*cos(theta)*sin(theta);
                    double BB = prediction[0].item<double>()*sin(theta)*sin(theta) + prediction[1].item<double>()*cos(theta)*cos(theta) - prediction[2].item<double>()*cos(theta)*sin(theta);
                    double theta1 = atan2(-test_out[0].item<double>()+test_out[1].item<double>()+sqrt(test_out[0].item<double>()*test_out[0].item<double>()-2*test_out[0].item<double>()*test_out[1].item<double>()+test_out[1].item<double>()*test_out[1].item<double>()+test_out[2].item<double>()*test_out[2].item<double>()),test_out[2].item<double>());
                    double AA1 = test_out[0].item<double>()*cos(theta1)*cos(theta1) + test_out[1].item<double>()*sin(theta1)*sin(theta1) + test_out[2].item<double>()*cos(theta1)*sin(theta1);
                    double BB1 = test_out[0].item<double>()*sin(theta1)*sin(theta1) + test_out[1].item<double>()*cos(theta1)*cos(theta1) - test_out[2].item<double>()*cos(theta1)*sin(theta1);
                    e2 = e2 + pow((4*AA*BB - 4*AA1*BB1),2.0)/pow(4*AA1*BB1,2.0);
                    //e2 = e2 + pow((4*prediction[0].item<double>()*prediction[1].item<double>() - 4*test_out[0].item<double>()*test_out[1].item<double>()),2.0);
                    //e2 = e2 + pow((4*prediction[0].item<double>()*prediction[1].item<double>() - 4*test_out[0].item<double>()*test_out[1].item<double>()),2.0)/pow(4*test_out[0].item<double>()*test_out[1].item<double>(),2.0);
                    e1 = e1 / size;
                    e = e + e1;
                    results_ex << "\n";
                    results_pr << "\n";
                }
                e = e / data_test.size().value();
                e2 = sqrt(e2 / data_test.size().value());
                std::cout << e2 << ","<< std::flush;
            }
            // else if (n == 7)
            // {
            //     for(int i = 0; i < data_test.size().value(); ++i)
            //     {
            //         test_in = data_test.get(i).data;
            //         test_out = data_test.get(i).target;

            //         IRL::LVIRANeighborhood<IRL::RectangularCuboid> neighborhood;
            //         neighborhood.resize(27);
            //         neighborhood.setCenterOfStencil(13);
            //         IRL::RectangularCuboid cells[27];
            //         for (int i = 0; i < 3; ++i) {
            //             for (int j = 0; j < 3; ++j) {
            //                 for (int k = 0; k < 3; ++k) {
            //                     double* a = new double();
            //                     *a = test_in[7*(i*9+j*3+k)].item<double>();
            //                     const double* b = a;
            //                         const int local_index =
            //                             (k) * 9 + (j) * 3 + (i);
            //                         cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
            //                             IRL::Pt(i-1.5, j-1.5, k-1.5),
            //                             IRL::Pt(i-0.5, j-0.5, k-0.5));
            //                         neighborhood.setMember(
            //                             static_cast<IRL::UnsignedIndex_t>(local_index),
            //                             &cells[local_index], b);
            //                         }
            //                     }
            //                 }
            //                 IRL::Pt a_gas_centroid = IRL::Pt(test_in[7*(1*9+1*3+1)+4].item<double>(), test_in[7*(1*9+1*3+1)+5].item<double>(), test_in[7*(1*9+1*3+1)+6].item<double>());
            //                 IRL::Pt a_liquid_centroid = IRL::Pt(test_in[7*(1*9+1*3+1)+1].item<double>(), test_in[7*(1*9+1*3+1)+2].item<double>(), test_in[7*(1*9+1*3+1)+3].item<double>());
            //                 auto bary_normal = IRL::Normal::fromPtNormalized(
            //                     a_gas_centroid - a_liquid_centroid);
            //                 bary_normal.normalize();
            //                 const double initial_distance =
            //                     bary_normal * neighborhood.getCenterCell().calculateCentroid();
            //                 IRL::PlanarSeparator a_interface = IRL::PlanarSeparator::fromOnePlane(
            //                     IRL::Plane(bary_normal, initial_distance));
            //                 IRL::setDistanceToMatchVolumeFractionPartialFill(
            //                     neighborhood.getCenterCell(),
            //                     neighborhood.getCenterCellStoredMoments(),
            //                     &a_interface);

            //                 a_interface =
            //                     IRL::reconstructionWithLVIRA3D(neighborhood, a_interface);
            //         IRL::Normal n = a_interface[0].normal();
            //         torch::Tensor prediction = torch::zeros(3);
            //         prediction[0] = -n[0];
            //         prediction[1] = -n[1];
            //         prediction[2] = -n[2];

            //         //auto prediction = nn->forward(test_in);
            //         auto loss = critereon_MSE(prediction, test_out);
            //         for (int j = 0; j < 3; ++j)
            //         {
            //             results_pr << prediction[j].item<double>() << " ";
            //         }
            //         for (int j = 0; j < 3; ++j)
            //         {
            //             results_ex << test_out[j].item<double>() << " ";
            //         }
            //         results_ex << "\n";
            //         results_pr << "\n";
            //     }
            // }

            // else if (n == 8)
            // {
            //     for(int i = 0; i < data_test.size().value(); ++i)
            //     {
            //         test_in = data_test.get(i).data;
            //         test_out = data_test.get(i).target;

            //         IRL::ELVIRANeighborhood neighborhood;
            //         neighborhood.resize(27);
            //         IRL::RectangularCuboid cells[27];
            //         for (int i = 0; i < 3; ++i) {
            //             for (int j = 0; j < 3; ++j) {
            //                 for (int k = 0; k < 3; ++k) {
            //                     double* a = new double();
            //                     *a = test_in[7*(i*9+j*3+k)].item<double>();
            //                     const double* b = a;
            //                         const int local_index =
            //                             (k) * 9 + (j) * 3 + (i);
            //                         cells[local_index] = IRL::RectangularCuboid::fromBoundingPts(
            //                             IRL::Pt(i-1.5, j-1.5, k-1.5),
            //                             IRL::Pt(i-0.5, j-0.5, k-0.5));
            //                         neighborhood.setMember(&cells[local_index], b,i-1,j-1,k-1);
            //                         }
            //                     }
            //                 }

            //                 IRL::PlanarSeparator a_interface =
            //                     IRL::reconstructionWithELVIRA3D(neighborhood);
            //         IRL::Normal n = a_interface[0].normal();
            //         torch::Tensor prediction = torch::zeros(3);
            //         prediction[0] = -n[0];
            //         prediction[1] = -n[1];
            //         prediction[2] = -n[2];

            //         //auto prediction = nn->forward(test_in);
            //         auto loss = critereon_MSE(prediction, test_out);
            //         for (int j = 0; j < 3; ++j)
            //         {
            //             results_pr << prediction[j].item<double>() << " ";
            //         }
            //         for (int j = 0; j < 3; ++j)
            //         {
            //             results_ex << test_out[j].item<double>() << " ";
            //         }
            //         results_ex << "\n";
            //         results_pr << "\n";
            //     }
            // }

           

            results_ex.close();
            results_pr.close();
        }
    }

    double trainer::test_model()
    {
        double e = 0;
        if (rank == 0)
        {
            auto data_test = MyDataset(test_in_file, test_out_file, data_size, type);
            // results_ex.open("result_ex.txt");
            // results_pr.open("result_pr.txt");
            //results.open("results.txt");
            int size = 0;
            nn->eval();
            size = nn->getOutput();
            for(int i = 0; i < data_test.size().value(); ++i)
            {
                double e1 = 0;
                test_in = data_test.get(i).data;
                test_out = data_test.get(i).target;
                torch::Tensor prediction = torch::zeros({size, 1});
                prediction = nn->forward(test_in);
                
                for (int j = 0; j < size; ++j)
                {
                    //results_pr << prediction[j].item<double>() << " ";
                    e1 = e1 + pow((prediction[j].item<double>() - test_out[j].item<double>()),2.0);
                }
                //for (int j = 0; j < size; ++j)
                {
                    //results_ex << test_out[j].item<double>() << " ";
                }
                e1 = e1 / size;
                e = e + e1;
                // results_ex << "\n";
                // results_pr << "\n";
            }
            e = e / data_test.size().value();

            // results_ex.close();
            // results_pr.close();
        }
        return e;
    }

    void trainer::load_model(std::string in)
    {
        torch::load(nn,in);
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

    // IRL::Normal trainer::get_para_axis(vector<double>* fractions)
    // {
    //     auto y_pred = nn->forward(torch::tensor(*fractions));
    //     auto n = IRL::Normal();
    //     n[0] = y_pred[0].item<double>();
    //     n[1] = y_pred[1].item<double>();
    //     n[2] = y_pred[2].item<double>();
    //     return n;
    // }

    vector<double> trainer::get_para_coeff(vector<double>* fractions)
    {
        auto y_pred = nn->forward(torch::tensor(*fractions));
        vector<double> coeff;
        coeff.push_back(y_pred[0].item<double>());
        coeff.push_back(y_pred[1].item<double>());
        coeff.push_back(y_pred[2].item<double>());
        coeff.push_back(y_pred[3].item<double>());
        coeff.push_back(y_pred[4].item<double>());
        return coeff;
    }

    vector<double> trainer::get_para_curvs(vector<double>* fractions)
    {
        auto y_pred = nn->forward(torch::tensor(*fractions));
        vector<double> coeff;
        coeff.push_back(y_pred[0].item<double>());
        coeff.push_back(y_pred[1].item<double>());
        return coeff;
    }

    // IRL::Normal trainer::get_para_origin(vector<double>* fractions)
    // {
    //     auto y_pred = nn->forward(torch::tensor(*fractions));
    //     auto n = IRL::Normal();
    //     n[0] = y_pred[0].item<double>();
    //     n[1] = y_pred[1].item<double>();
    //     n[2] = y_pred[2].item<double>();
    //     return n;
    // }

    double trainer::get_curv(vector<double>* fractions)
    {
        auto y_pred = nn->forward(torch::tensor(*fractions));
        return y_pred[0].item<double>();
    }

    double trainer::get_type(vector<double>* fractions)
    {
        auto y_pred = nn->forward(torch::tensor(*fractions));
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















   cnn_trainer::cnn_trainer(int s)
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

    cnn_trainer::cnn_trainer(int e, int d, double l, int s)
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
    
    cnn_trainer::~cnn_trainer()
    {
        delete optimizer;
    }

    void cnn_trainer::init()
    {
        nn_cnn = make_shared<model_cnn>(5,3,100);
        optimizer = new torch::optim::Adam(nn_cnn->parameters(), learning_rate);
        critereon_BCE = torch::nn::BCELoss();
        critereon_MSE = torch::nn::MSELoss();
    }

    void cnn_trainer::load_train_data(string in_file, string out_file)
    {
        train_in_file = in_file;
        train_out_file = out_file;
    }

    void cnn_trainer::load_validation_data(string in_file, string out_file, int x)
    {
        validation_in_file = in_file;
        validation_out_file = out_file;
        data_val_size = x;
    }

    void cnn_trainer::load_test_data(string in_file, string out_file)
    {
        test_in_file = in_file;
        test_out_file = out_file;
    }

    void cnn_trainer::train_model(bool load, std::string in, std::string out)
    {
        cout << "Hello from rank " << rank << endl;
        auto data_train = MyDataset_cnn(train_in_file, train_out_file, data_size, type).map(torch::data::transforms::Stack<>());
        batch_size = data_train.size().value() / numranks;
        if (rank == 0)
        {
            cout << data_size << " " << batch_size << endl;
        }
        auto data_sampler = torch::data::samplers::DistributedRandomSampler(data_train.size().value(), numranks, rank, false);
        auto data_loader_train = torch::data::make_data_loader(std::move(data_train), data_sampler, batch_size);

        auto data_val = MyDataset_cnn(validation_in_file, validation_out_file, data_val_size, type).map(torch::data::transforms::Stack<>());
        val_batch_size = data_val.size().value() / numranks;
        double val_size = data_val.size().value() / numranks;
        auto data_sampler_val = torch::data::samplers::DistributedRandomSampler(data_val.size().value(), numranks, rank, false);
        auto data_loader_val = torch::data::make_data_loader(std::move(data_val), data_sampler_val, val_size);

        if (load)
        {
            torch::load(nn_cnn, in);
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

            nn_cnn->train();

            for (auto& batch : *data_loader_train)
            {
                train_in = batch.data;
                train_out = batch.target;

                torch::Tensor y_pred = torch::zeros({batch_size, size});
                torch::Tensor check;
                torch::Tensor comp;

                y_pred = nn_cnn->forward(train_in);
                check = y_pred;
                comp = train_out;

                torch::Tensor loss = torch::zeros({batch_size, 1});
                torch::Tensor l2 = torch::tensor(0.0);
                torch::Tensor l1 = torch::tensor(0.0);
                for (auto &param : nn_cnn->named_parameters())
                {
                    if (param.key().find("weight") != std::string::npos)
                    {
                        l2 = l2 + lambda2 * param.value().square().sum();
                        l1 = l1 + lambda1 * param.value().abs().sum();
                    }
                }

                //loss = critereon_BCE(check, comp) + l2 + l1;
                loss = critereon_MSE(check.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)}), comp.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)})) + functions->MSE_paraboloid_ref(check,comp) + l2 + l1;
                count = 0;
                // for (int i = 0; i < batch_size; ++i)
                // {
                //     if ((comp[i].item<double>() == 1 && check[i].item<double>() > 0.5) || (comp[i].item<double>() == 0 && check[i].item<double>() <= 0.5))
                //     {
                //         ++count;
                //     }
                // }
                epoch_loss = epoch_loss + loss.item().toDouble()*batch.data.size(0);

                optimizer->zero_grad();
                loss.backward();

                if (numranks > 1)
                {
                    for (auto &param : nn_cnn->named_parameters())
                    {
                        MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(), param.value().grad().numel(), mpiDatatype.at(param.value().grad().scalar_type()), MPI_SUM, MPI_COMM_WORLD);
                        param.value().grad().data() = param.value().grad().data()/numranks;
                        
                    } 
                    MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                }   

                optimizer->step();
            }

            nn_cnn->eval();             
            
            for (auto& batch : *data_loader_val)
            {
                val_in = batch.data;
                val_out = batch.target;
                torch::Tensor y_pred = torch::zeros({val_batch_size, size});
                torch::Tensor check;
                torch::Tensor comp;

                y_pred = nn_cnn->forward(val_in);
                check = y_pred;
                comp = val_out;

                torch::Tensor loss = torch::zeros({val_batch_size, 1});
                torch::Tensor l2 = torch::tensor(0.0);
                torch::Tensor l1 = torch::tensor(0.0);
                for (auto &param : nn_cnn->named_parameters())
                {
                    if (param.key().find("weight") != std::string::npos)
                    {
                        l2 = l2 + lambda2 * param.value().square().sum();
                        l1 = l1 + lambda1 * param.value().abs().sum();
                    }
                }
                //loss = critereon_BCE(check, comp) + l2 + l1;
                loss = critereon_MSE(check.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)}), comp.index({torch::indexing::Slice(),torch::indexing::Slice(0,2)})) + functions->MSE_paraboloid_ref(check,comp) + l2 + l1;
                epoch_loss_val = epoch_loss_val + loss.item().toDouble()*batch.data.size(0);

                if (numranks > 1)
                {
                    MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
                //cout << epoch << " " << count << "/" << batch_size << endl;
                cout << epoch << " " << total_epoch_loss << " " << total_epoch_loss_val << endl;
                std::cout.flush();
            }
            if (epoch % 100 == 0)
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
            if (rank == 0 && (epoch % 100 == 0 || epoch == epochs - 1))
            {
                torch::save(nn_cnn, out);                   
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Finalize(); 
    }

    void cnn_trainer::test_model(int n)
    {
        if (rank == 0)
        {
            auto data_test = MyDataset_cnn(test_in_file, test_out_file, data_size, type);
            results_ex.open("result_ex.txt");
            results_pr.open("result_pr.txt");
            int size = 0;

            nn_cnn->eval();
            size = 5;
            int count = 0;
            int total = data_test.size().value();
            for(int i = 0; i < data_test.size().value(); ++i)
            {
                test_in = torch::zeros({1,7,3,3,3});
                test_in[0] = data_test.get(i).data;
                test_out = data_test.get(i).target;
                torch::Tensor prediction = nn_cnn->forward(test_in);
                // if ((test_out[0].item<double>() == 1 && prediction[0].item<double>() > 0.5) || (test_out[0].item<double>() == 0 && prediction[0].item<double>() <= 0.5))
                // {
                //     ++count;
                // }

                for (int j = 0; j < size; ++j)
                {
                    results_pr << prediction[0][j].item<double>() << " ";
                }
                for (int j = 0; j < size; ++j)
                {
                    results_ex << test_out[j].item<double>() << " ";
                }
                results_ex << "\n";
                results_pr << "\n";
            }
            //std::cout << "Result: " << count << "/" << total << " (" << data_test.size().value() << ")" << std::endl;
        }
    }

    void cnn_trainer::load_model(std::string in)
    {
        torch::load(nn_cnn,in);
    }

    double cnn_trainer::get_type(vector<double>* fractions)
    {
        auto y_pred = nn_cnn->forward(torch::tensor(*fractions));
        return y_pred[0].item<double>();
    }

}

#endif