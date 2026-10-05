// // This file is part of the Interface Reconstruction Library (IRL),
// // a library for interface reconstruction and computational geometry operations.
// //
// // Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
// //
// // This Source Code Form is subject to the terms of the Mozilla Public
// // License, v. 2.0. If a copy of the MPL was not distributed with this
// // file, You can obtain one at https://mozilla.org/MPL/2.0/.

// #include "irl/machine_learning_reconstruction/trainer.h"

// namespace IRL
// {
//     trainer::trainer(int s)
//     {
//         rank = 0;
//         numranks = 1;                      
//         epochs = 0;
//         data_size = 0;
//         batch_size = 0;
//         learning_rate = 0.001;
//         type = s;
//         init();
//     }

//     trainer::trainer(int e, int d, double l, int s)
//     {
//         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//         MPI_Comm_size(MPI_COMM_WORLD, &numranks);                    
//         epochs = e;
//         data_size = d;
//         batch_size = data_size / numranks;
//         learning_rate = l;
//         type = s;
//         init();
//     }

//     trainer::trainer(int e, int d, int nh, int h, double l, int s)
//     {
//         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//         MPI_Comm_size(MPI_COMM_WORLD, &numranks);                    
//         epochs = e;
//         data_size = d;
//         batch_size = data_size / numranks;
//         learning_rate = l;
//         type = s;
//         nn = std::make_shared<model>(189,3,nh,h,0);
//         optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
//         critereon_MSE = torch::nn::MSELoss();
//         std::cout.precision(15);
//     }

//     void trainer::init()
//     {
//         std::cout.precision(15);
//         switch (type)
//         {
//             case 0:
//                 nn = std::make_shared<model>(189,3,3,100,0);
//                 optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
//                 critereon_MSE = torch::nn::MSELoss();
//             break;
//             case 1:
//                 nn = std::make_shared<model>(192,6,3,100,1);
//                 optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
//                 critereon_MSE = torch::nn::MSELoss();
//             break;
//             case 2:
//                 nn = std::make_shared<model>(189,1,3,100,2);
//                 optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
//                 critereon_MSE = torch::nn::MSELoss();
//                 critereon_BCE = torch::nn::BCELoss();
//             break;
//             case 3:
//                 nn = std::make_shared<model>(189,3,5,100,3);
//                 optimizer = std::make_unique<torch::optim::Adam>(nn->parameters(), learning_rate);
//                 //optimizer = new torch::optim::Adam(nn->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(0.001));
//                 critereon_MSE = torch::nn::MSELoss();
//                 critereon_BCE = torch::nn::BCELoss();
//                 critereon_CE = torch::nn::CrossEntropyLoss();
//             break;
//         }
//     }

//     void trainer::load_train_data(std::string in_file, std::string out_file)
//     {
//         train_in_file = in_file;
//         train_out_file = out_file;
//     }

//     void trainer::load_validation_data(std::string in_file, std::string out_file, int x)
//     {
//         validation_in_file = in_file;
//         validation_out_file = out_file;
//         data_val_size = x;
//     }

//     void trainer::load_test_data(std::string in_file, std::string out_file)
//     {
//         test_in_file = in_file;
//         test_out_file = out_file;
//     }

//     void trainer::train_model(bool load, std::string in, std::string out)
//     {
//         std::cout << "Hello from rank " << rank << std::endl;
//         auto data_train = MyDataset(train_in_file, train_out_file, data_size).map(torch::data::transforms::Stack<>());
//         batch_size = data_train.size().value() / numranks;
//         if (rank == 0)
//         {
//             std::cout << data_size << " " << batch_size << std::endl;
//         }
//         auto data_sampler = torch::data::samplers::DistributedRandomSampler(data_train.size().value(), numranks, rank, false);
//         auto data_loader_train = torch::data::make_data_loader(std::move(data_train), data_sampler, batch_size);

//         auto data_val = MyDataset(validation_in_file, validation_out_file, data_val_size).map(torch::data::transforms::Stack<>());
//         val_batch_size = data_val.size().value() / numranks;
//         int val_size = data_val.size().value() / numranks;
//         auto data_sampler_val = torch::data::samplers::DistributedRandomSampler(data_val.size().value(), numranks, rank, false);
//         auto data_loader_val = torch::data::make_data_loader(std::move(data_val), data_sampler_val, val_size);

//         if (load)
//         {
//             torch::load(nn, in);
//         }
//         double epoch_loss_val_check = 0;
//         double lambda1 = 0.0;
//         double lambda2 = 0.0;
//         for (int epoch = 0; epoch < epochs; ++epoch)
//         {
//             double epoch_loss = 0;
//             double epoch_loss_val = 0;
//             double total_epoch_loss = 0;
//             double total_epoch_loss_val = 0;
//             int count = 0;
//             int size = 0;

//             nn->train();
//             size = nn->getOutput();

//             for (auto& batch : *data_loader_train)
//             {
//                 train_in = batch.data;
//                 train_out = batch.target;

//                 torch::Tensor y_pred = torch::zeros({batch_size, size});
//                 torch::Tensor check;
//                 torch::Tensor comp;

//                 y_pred = nn->forward(train_in);

//                 check = y_pred;
//                 comp = train_out;

//                 torch::Tensor loss = torch::zeros({batch_size, 1});
//                 torch::Tensor l2 = torch::tensor(0.0);
//                 torch::Tensor l1 = torch::tensor(0.0);
//                 for (auto &param : nn->named_parameters())
//                 {
//                     if (param.key().find("weight") != std::string::npos)
//                     {
//                         l2 = l2 + lambda2 * param.value().square().sum();
//                         l1 = l1 + lambda1 * param.value().abs().sum();
//                     }
//                 }
//                 if (type == 1)
//                 {
//                     auto loss_i = torch::nn::functional::mse_loss(check, comp, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone));
//                     auto p1 = check.narrow(-1, 0, 3);
//                     auto p2 = check.narrow(-1, 3, 3);
//                     auto t1 = comp.narrow(-1, 0, 3);
//                     auto t2 = comp.narrow(-1, 3, 3);
//                     auto deadzone_v1 = (t1.norm(2, -1, true) < 0.1) & (p1.norm(2, -1, true) < 0.8);
//                     auto deadzone_v2 = (t2.norm(2, -1, true) < 0.1) & (p2.norm(2, -1, true) < 0.8);
//                     auto loss_v1 = loss_i.narrow(-1, 0, 3).masked_fill(deadzone_v1.expand_as(p1), 0.0);
//                     auto loss_v2 = loss_i.narrow(-1, 3, 3).masked_fill(deadzone_v2.expand_as(p2), 0.0);
//                     loss = torch::cat({loss_v1, loss_v2}, -1).mean();
//                 }
//                 else if (type == 2)
//                 {
//                     loss = critereon_BCE(check, comp) + l2 + l1;
//                     count += ((check > 0.5) == comp).sum().item<int>();
//                 }
//                 else if (type == 3)
//                 {
//                     auto target_classes = torch::argmax(comp, 1);
//                     loss = critereon_CE(check, target_classes) + l2 + l1;
//                     count += torch::argmax(check, 1).eq(target_classes).sum().item<int>();
//                 }
//                 else
//                 {
//                     loss = critereon_MSE(check, comp);
//                 }
//                 epoch_loss = epoch_loss + loss.item().toDouble()*batch.data.size(0);

//                 optimizer->zero_grad();
//                 loss.backward();

//                 if (numranks > 1)
//                 {
//                     for (auto &param : nn->named_parameters())
//                     {
//                         MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(), param.value().grad().numel(), mpiDatatype.at(param.value().grad().scalar_type()), MPI_SUM, MPI_COMM_WORLD);
//                         param.value().grad().div_(numranks);
                        
//                     } 
//                 }

//                 optimizer->step();
//             }

//             if (numranks > 1)
//             {
//                 MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//             }

//             nn->eval();    
            
//             for (auto& batch : *data_loader_val)
//             {
//                 val_in = batch.data;
//                 val_out = batch.target;
//                 torch::Tensor y_pred = torch::zeros({val_batch_size, size});
//                 torch::Tensor check;
//                 torch::Tensor comp;

//                 y_pred = nn->forward(val_in);

//                 check = y_pred;
//                 comp = val_out;

//                 torch::Tensor loss = torch::zeros({val_batch_size, 1});
//                 torch::Tensor l2 = torch::tensor(0.0);
//                 torch::Tensor l1 = torch::tensor(0.0);
//                 for (auto &param : nn->named_parameters())
//                 {
//                     if (param.key().find("weight") != std::string::npos)
//                     {
//                         l2 = l2 + lambda2 * param.value().square().sum();
//                         l1 = l1 + lambda1 * param.value().abs().sum();
//                     }
//                 }
//                 if (type == 1)
//                 {
//                     auto loss_i = torch::nn::functional::mse_loss(check, comp, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone));
//                     auto p1 = check.narrow(-1, 0, 3);
//                     auto p2 = check.narrow(-1, 3, 3);
//                     auto t1 = comp.narrow(-1, 0, 3);
//                     auto t2 = comp.narrow(-1, 3, 3);
//                     auto deadzone_v1 = (t1.norm(2, -1, true) < 0.1) & (p1.norm(2, -1, true) < 0.8);
//                     auto deadzone_v2 = (t2.norm(2, -1, true) < 0.1) & (p2.norm(2, -1, true) < 0.8);
//                     auto loss_v1 = loss_i.narrow(-1, 0, 3).masked_fill(deadzone_v1.expand_as(p1), 0.0);
//                     auto loss_v2 = loss_i.narrow(-1, 3, 3).masked_fill(deadzone_v2.expand_as(p2), 0.0);
//                     loss = torch::cat({loss_v1, loss_v2}, -1).mean();
//                 }
//                 else if (type == 2)
//                 {
//                     loss = critereon_BCE(check, comp) + l2 + l1;
//                 }
//                 else if (type == 3)
//                 {
//                     auto target_classes = torch::argmax(comp, 1);
//                     loss = critereon_CE(check, target_classes) + l2 + l1;
//                 }
//                 else
//                 {
//                     loss = critereon_MSE(check, comp);
//                 }
//                 epoch_loss_val = epoch_loss_val + loss.item().toDouble()*batch.data.size(0);
//             }
//             if (numranks == 1)
//             {
//                 total_epoch_loss = epoch_loss;
//                 total_epoch_loss_val = epoch_loss_val;
//             }
//             else
//             {
//                 MPI_Allreduce(&epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//                 MPI_Allreduce(&epoch_loss_val, &total_epoch_loss_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//             }
//             total_epoch_loss = total_epoch_loss / data_size;
//             total_epoch_loss_val = total_epoch_loss_val / data_val_size;
//             if (rank == 0)
//             {
//                 if (type == 2 || type == 3)
//                 {
//                     std::cout << epoch << " " << count << "/" << batch_size << std::endl;
//                 }
//                 std::cout << epoch << " " << total_epoch_loss << " " << total_epoch_loss_val << std::endl;
//                 std::cout.flush();
//             }
//             if (epoch % 100 == 0)
//             {
//                 if (total_epoch_loss_val < epoch_loss_val_check || epoch == 0)
//                 {
//                     epoch_loss_val_check = total_epoch_loss_val;
//                 }
//                 else if (epoch < epochs-1)
//                 {
//                     epoch = epochs;
//                 }
//             }
            
//             MPI_Barrier(MPI_COMM_WORLD);
//             if (rank == 0 && (epoch % 100 == 0 || epoch == epochs - 1))
//             {
//                 torch::save(nn, out);
//             }
//             MPI_Barrier(MPI_COMM_WORLD);
//         } 
//     }

//     // VF-weight the barycentre entries of a raw-189 stencil to match plicnet_v2's
//     // training inputs. Layout: 27 cells x 7 -> [VF, bx,by,bz, gx,gy,gz] per cell.
//     // No-op if the data was generated with normalize=false.
//     static torch::Tensor vf_weight_moments(const torch::Tensor& in)
//     {
//         auto out = in.clone().to(torch::kDouble).view({-1, 7});   // 27 x 7
//         auto vf  = out.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)});
//         out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, 7)},
//                     out.index({torch::indexing::Slice(), torch::indexing::Slice(1, 7)}) * vf);
//         return out.view({-1});                                    // 189
//     }

//     void trainer::test_model(std::string ex, std::string pr)
//     {
//         if (rank == 0)
//         {
//             auto data_test = MyDataset(test_in_file, test_out_file, data_size);
//             results_ex.open(ex);
//             results_pr.open(pr);
//             int size = 0;

//             if (type == 2)
//             {
//                 nn->eval();
//                 size = 1;
//                 int count = 0;
//                 int total = data_test.size().value();
//                 for(int i = 0; i < data_test.size().value(); ++i)
//                 {
//                     test_in = data_test.get(i).data;
//                     test_out = data_test.get(i).target;
//                     torch::Tensor prediction = torch::zeros({1, 1});
//                     prediction = nn->forward(test_in.unsqueeze(0));
//                     if ((test_out[0].item<double>() == 1 && prediction[0].item<double>() > 0.5) || (test_out[0].item<double>() == 0 && prediction[0].item<double>() <= 0.5))
//                     {
//                         ++count;
//                     }

//                     results_pr << prediction[0].item<double>();
//                     results_ex << test_out[0].item<double>();

//                     results_ex << "\n";
//                     results_pr << "\n";
//                 }
//                 std::cout << "Result: " << count << "/" << total << " (" << data_test.size().value() << ")" << std::endl;
//             }
//             else if (type == 3)
//             {
//                 nn->eval();
//                 size = 3;
//                 int count = 0;
//                 int total = data_test.size().value();
//                 for(int i = 0; i < data_test.size().value(); ++i)
//                 {
//                     test_in = data_test.get(i).data;
//                     test_out = data_test.get(i).target;
//                     torch::Tensor prediction = torch::softmax(nn->forward(test_in.unsqueeze(0)), 1).squeeze(0);
//                     auto ind = torch::argmax(test_out);
//                     auto ind2 = torch::argmax(prediction);
//                     if (ind.item<int>() == ind2.item<int>())
//                     {
//                         ++count;
//                     }

//                     for (int j = 0; j < size; ++j)
//                     {
//                         results_pr << prediction[j].item<double>() << " ";
//                     }
//                     for (int j = 0; j < size; ++j)
//                     {
//                         results_ex << test_out[j].item<double>() << " ";
//                     }

//                     results_ex << "\n";
//                     results_pr << "\n";
//                 }
//                 std::cout << "Result: " << count << "/" << total << " (" << data_test.size().value() << ")" << std::endl;
//             }
//             else
//             {
//                 torch::NoGradGuard no_grad;

//                 // if (use_jit)
//                 // {
//                 //     jit_nn.eval();
//                 //     size = 3;                 // scripted module has no getOutput()
//                 // }
//                 // else
//                 {
//                     nn->eval();
//                     size = nn->getOutput();
//                 }

//                 for(int i = 0; i < data_test.size().value(); ++i)
//                 {
//                     test_in = data_test.get(i).data;
//                     test_out = data_test.get(i).target;

//                     // torch::Tensor net_in = vf_weighted_inputs
//                     //     ? vf_weight_moments(test_in)
//                     //     : test_in;
//                     // net_in = net_in.to(test_in.scalar_type());

//                     // torch::Tensor prediction = use_jit
//                     //     ? jit_nn.forward({net_in.unsqueeze(0)}).toTensor().squeeze(0)
//                     //     : nn->forward(net_in.unsqueeze(0)).squeeze(0);
//                     torch::Tensor prediction = nn->forward(test_in.unsqueeze(0)).squeeze(0);

//                     for (int j = 0; j < size; ++j)
//                     {
//                         results_pr << prediction[j].item<double>() << " ";
//                     }
//                     for (int j = 0; j < size; ++j)
//                     {
//                         results_ex << test_out[j].item<double>() << " ";
//                     }
//                     results_ex << "\n";
//                     results_pr << "\n";
//                 }
//             }
           
//             results_ex.close();
//             results_pr.close();
//         }
//     }

// IRL::Normal trainer::predict_normal(const torch::Tensor& stencil_189)
//     {
//         torch::NoGradGuard no_grad;
//         nn->eval();

//         // Match test_model's active branch exactly: the network was trained
//         // on raw-189 canonicalized moments, so no VF weighting here. If the
//         // commented use_jit / vf_weighted_inputs branches in test_model and
//         // load_model are ever re-enabled, mirror them here too.
//         torch::Tensor in = stencil_189.to(torch::kFloat32);
//         torch::Tensor out = nn->forward(in.unsqueeze(0)).squeeze(0);

//         IRL::Normal n(out[0].item<double>(), out[1].item<double>(),
//                       out[2].item<double>());
//         if (IRL::squaredMagnitude(n) < 1.0e-24) return IRL::Normal(0.0, 0.0, 1.0);
//         n.normalize();
//         return n;
//     }

//     void trainer::load_model(std::string in)
//     {
//         // // plicnet_v2.pt is TorchScript and expects VF-weighted first moments;
//         // // the dropin and legacy NN1 archives take raw-189 barycentres.
//         // if (in.size() >= 3 && in.compare(in.size() - 3, 3, ".pt") == 0 &&
//         //     in.find("dropin") == std::string::npos)
//         // {
//         //     jit_nn = torch::jit::load(in);
//         //     jit_nn.eval();
//         //     use_jit = true;
//         //     vf_weighted_inputs = true;
//         // }
//         // else
//         {
//             torch::load(nn, in);
//             //nn->eval();
//             use_jit = false;
//             vf_weighted_inputs = false;
//         }
//         model_path = in;
//     }
// }

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
    // Permutation-invariant MSE for the two-plane (type == 1) model.
    // pred/target are [B, 6] = concatenated (n1, n2), each a 3-vector.
    //
    // (n1, n2) and (n2, n1) describe the same physical reconstruction, and the
    // generator no longer canonicalizes their order (the flip_normals block in
    // data_gen.h was discontinuous: two nearly identical films straddling
    // normal1[0] == 0 got swapped-and-negated labels). So the loss takes the
    // minimum squared error over the two possible pairings per sample:
    //   L = min( |p1-t1|^2 + |p2-t2|^2,  |p1-t2|^2 + |p2-t1|^2 )
    //
    // Deliberately NOT normalized. Normalizing the predictions makes the loss
    // scale-invariant in pred, which leaves no gradient acting on magnitude:
    // the outputs drift toward zero, and the normalization Jacobian diverges
    // like 1/|p| exactly there, which is the "small vectors, all alike"
    // collapse. Targets on disk are already unit vectors, so plain squared
    // error constrains direction and magnitude together, exactly as the
    // previous MSE did -- this changes only the pairing, nothing else.
    torch::Tensor permutation_invariant_normal_loss(const torch::Tensor& pred, const torch::Tensor& target)
    {
        auto p1 = pred.narrow(-1, 0, 3);
        auto p2 = pred.narrow(-1, 3, 3);
        auto t1 = target.narrow(-1, 0, 3);
        auto t2 = target.narrow(-1, 3, 3);

        auto sq = [](const torch::Tensor& a, const torch::Tensor& b) {
            return (a - b).pow(2).sum(-1);   // [B]
        };

        auto cost_straight = sq(p1, t1) + sq(p2, t2);
        auto cost_swapped  = sq(p1, t2) + sq(p2, t1);

        // Divide by 6 so the scale matches a plain mse_loss over all six
        // components, keeping loss values comparable to the previous runs.
        auto loss_per_sample = torch::min(cost_straight, cost_swapped) / 6.0;
        return loss_per_sample.mean();
    }

    // ---------------------------------------------------------------------
    // Gradient fusion.
    //
    // The previous implementation issued one MPI_Allreduce per parameter
    // tensor per step. With a small MLP that is ~8 latency-bound collectives
    // for a few tens of thousands of doubles -- fine when there was one step
    // per epoch, but the dominant cost once batching produces hundreds of
    // steps per epoch. Here all gradients are packed into one contiguous
    // buffer, reduced with a single collective, and scattered back.
    // ---------------------------------------------------------------------
    void trainer::build_grad_buffer()
    {
        grad_params.clear();
        long total = 0;
        // named_parameters() has a deterministic order and every rank builds
        // an identical model, so the packing order is consistent across ranks.
        for (auto &param : nn->named_parameters())
        {
            grad_params.push_back(param.value());
            total += param.value().numel();
        }
        auto opts = torch::TensorOptions()
                        .dtype(grad_params.empty() ? torch::kFloat32
                                                   : grad_params[0].scalar_type())
                        .device(grad_params.empty() ? torch::kCPU
                                                    : grad_params[0].device());
        grad_flat_buffer = torch::zeros({total}, opts);
    }

    void trainer::allreduce_gradients()
    {
        if (numranks <= 1) return;

        // Pack. Parameters with no gradient this step contribute zeros, which
        // is the correct contribution to a summed gradient and avoids the
        // undefined-tensor deref the old per-parameter loop could hit.
        long offset = 0;
        for (auto &p : grad_params)
        {
            long n = p.numel();
            if (p.grad().defined())
            {
                grad_flat_buffer.narrow(0, offset, n).copy_(p.grad().reshape({n}));
            }
            else
            {
                grad_flat_buffer.narrow(0, offset, n).zero_();
            }
            offset += n;
        }

        MPI_Allreduce(MPI_IN_PLACE,
                      grad_flat_buffer.data_ptr(),
                      grad_flat_buffer.numel(),
                      mpiDatatype.at(grad_flat_buffer.scalar_type()),
                      MPI_SUM, MPI_COMM_WORLD);
        grad_flat_buffer.div_(numranks);

        // Unpack.
        offset = 0;
        for (auto &p : grad_params)
        {
            long n = p.numel();
            if (p.grad().defined())
            {
                p.grad().copy_(grad_flat_buffer.narrow(0, offset, n).reshape(p.sizes()));
            }
            offset += n;
        }
    }

    // ---------------------------------------------------------------------
    // LR schedule: linear warmup then cosine decay.
    //
    // Warmup matters here because the effective batch is batch_size*numranks,
    // which is large even for modest per-rank batches; stepping straight in at
    // a batch-scaled LR is the usual way large-batch runs diverge early.
    // ---------------------------------------------------------------------
    double trainer::lr_at_step(long step, long total_steps) const
    {
        double peak = base_learning_rate;
        if (scale_lr_with_batch)
        {
            // sqrt scaling relative to a 256-sample reference global batch.
            // Linear scaling is the other common choice and is more aggressive;
            // sqrt is the safer default for a regression objective like this.
            double global_batch = static_cast<double>(batch_size) * numranks;
            peak = base_learning_rate * std::sqrt(global_batch / 256.0);
        }

        if (step < warmup_steps && warmup_steps > 0)
        {
            return peak * (static_cast<double>(step) + 1.0) / warmup_steps;
        }

        // Plateau mode: hold the peak LR and let the validation callback
        // shrink plateau_lr_scale when progress stalls. No horizon needed.
        if (use_plateau_decay)
        {
            return peak * std::max(plateau_lr_scale, plateau_min_lr_factor);
        }

        long decay_steps = std::max(1L, total_steps - warmup_steps);
        double progress = static_cast<double>(step - warmup_steps) / decay_steps;
        progress = std::min(1.0, std::max(0.0, progress));
        double cosine = 0.5 * (1.0 + std::cos(M_PI * progress));
        return peak * (min_lr_factor + (1.0 - min_lr_factor) * cosine);
    }

    trainer::trainer(int s)
    {
        rank = 0;
        numranks = 1;                      
        epochs = 0;
        data_size = 0;
        // batch_size intentionally left at the header default; this
        // constructor is used for inference, and zero would be a division
        // by zero in train_model's step-count computation.
        learning_rate = 0.001;
        base_learning_rate = learning_rate;
        type = s;
        init();
    }

    trainer::trainer(int e, int d, double l, int s)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numranks);                    
        epochs = e;
        data_size = d;
        // NOTE: batch_size is deliberately NOT set to data_size/numranks here.
        // Gradients are allreduced every step, so that choice makes the global
        // batch the entire dataset and yields one optimizer step per epoch.
        // It keeps the header default; override with set_batch_size().
        learning_rate = l;
        base_learning_rate = learning_rate;
        type = s;
        init();
    }

    trainer::trainer(int e, int d, int nh, int h, double l, int s)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numranks);                    
        epochs = e;
        data_size = d;
        // NOTE: batch_size is deliberately NOT set to data_size/numranks here.
        // Gradients are allreduced every step, so that choice makes the global
        // batch the entire dataset and yields one optimizer step per epoch.
        // It keeps the header default; override with set_batch_size().
        learning_rate = l;
        base_learning_rate = learning_rate;
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
                nn = std::make_shared<model>(192,6,3,256,1);
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
        if (batch_size <= 0 || val_batch_size <= 0)
        {
            if (rank == 0)
            {
                std::cerr << "trainer: batch_size and val_batch_size must be > 0 "
                          << "(got " << batch_size << ", " << val_batch_size << ")"
                          << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const double lambda1 = 0.0;
        const double lambda2 = 0.0;
        const bool use_reg = (lambda1 != 0.0 || lambda2 != 0.0);

        auto data_train = MyDataset(train_in_file, train_out_file, data_size)
                              .map(torch::data::transforms::Stack<>());
        auto data_val = MyDataset(validation_in_file, validation_out_file, data_val_size)
                            .map(torch::data::transforms::Stack<>());

        const long train_total = data_train.size().value();
        const long val_total   = data_val.size().value();

        // Each rank draws a disjoint shard via the distributed sampler, then
        // iterates that shard in minibatches of batch_size. Gradients are
        // averaged across ranks every step, so the effective global batch is
        // batch_size * numranks.
        auto data_sampler = torch::data::samplers::DistributedRandomSampler(
            train_total, numranks, rank, false);
        auto data_loader_train = torch::data::make_data_loader(
            std::move(data_train), data_sampler, batch_size);

        auto data_sampler_val = torch::data::samplers::DistributedRandomSampler(
            val_total, numranks, rank, false);
        auto data_loader_val = torch::data::make_data_loader(
            std::move(data_val), data_sampler_val, val_batch_size);

        if (load)
        {
            torch::load(nn, in);
        }

        build_grad_buffer();

        // -----------------------------------------------------------------
        // Determine a step count every rank agrees on.
        //
        // DistributedRandomSampler with allow_duplicates=false can hand
        // different ranks a different number of samples when the dataset does
        // not divide evenly, which means different batch counts. Since every
        // step contains a collective, a rank that runs one extra batch will
        // hang in MPI_Allreduce while the others have moved on. So take the
        // MINIMUM batch count across ranks and stop there; the few leftover
        // samples on longer ranks are simply reshuffled into the next epoch.
        // -----------------------------------------------------------------
        long local_batches = (train_total / numranks + batch_size - 1) / batch_size;
        long steps_per_epoch = local_batches;
        if (numranks > 1)
        {
            MPI_Allreduce(MPI_IN_PLACE, &steps_per_epoch, 1, MPI_LONG,
                          MPI_MIN, MPI_COMM_WORLD);
        }
        steps_per_epoch = std::max(1L, steps_per_epoch);

        long local_val_batches = (val_total / numranks + val_batch_size - 1) / val_batch_size;
        long val_steps = local_val_batches;
        if (numranks > 1)
        {
            MPI_Allreduce(MPI_IN_PLACE, &val_steps, 1, MPI_LONG,
                          MPI_MIN, MPI_COMM_WORLD);
        }
        val_steps = std::max(1L, val_steps);

        // Cosine horizon. Defaults to the full run length, but that is only
        // correct if the run actually goes the distance; when epochs is a
        // generous upper bound with early stopping, set schedule_steps to
        // where convergence is expected or the decay never engages.
        const long total_steps = (schedule_steps > 0)
                                     ? schedule_steps
                                     : steps_per_epoch * epochs;
        const long max_steps = steps_per_epoch * epochs;

        if (rank == 0)
        {
            std::cout << "# train samples      " << train_total << std::endl;
            std::cout << "# ranks              " << numranks << std::endl;
            std::cout << "# per-rank batch     " << batch_size << std::endl;
            std::cout << "# global batch       " << (long)batch_size * numranks << std::endl;
            std::cout << "# steps/epoch        " << steps_per_epoch << std::endl;
            std::cout << "# max steps          " << max_steps << std::endl;
            if (use_plateau_decay)
            {
                std::cout << "# lr schedule        plateau (x" << plateau_factor
                          << " after " << plateau_patience << " stalled evals)" << std::endl;
                std::cout << "# peak lr            " << lr_at_step(warmup_steps, total_steps) << std::endl;
            }
            else
            {
                std::cout << "# lr schedule        cosine over " << total_steps << " steps"
                          << (schedule_steps > 0 ? " (explicit)" : " (= steps/epoch * epochs)")
                          << std::endl;
                std::cout << "# peak lr            " << lr_at_step(warmup_steps, total_steps) << std::endl;
                std::cout << "# final lr           " << lr_at_step(total_steps, total_steps) << std::endl;
            }
            std::cout << "# step train_loss val_loss lr" << std::endl;
            std::cout.flush();
        }

        double best_val = std::numeric_limits<double>::max();
        int evals_without_improvement = 0;
        bool stop = false;
        global_step = 0;

        for (int epoch = 0; epoch < epochs && !stop; ++epoch)
        {
            nn->train();

            // Accumulate on-device and sync once per logging window rather
            // than calling .item() every step, which would serialize the
            // whole pipeline on a device transfer at every iteration.
            torch::Tensor running_loss = torch::zeros({1}, torch::kFloat64);
            long running_count = 0;
            long running_correct = 0;
            long batch_index = 0;

            for (auto& batch : *data_loader_train)
            {
                if (batch_index >= steps_per_epoch) break;   // keep ranks in lockstep
                ++batch_index;

                auto check = nn->forward(batch.data);
                auto comp  = batch.target;

                torch::Tensor loss;
                torch::Tensor reg = torch::zeros({1}, check.options());
                if (use_reg)
                {
                    for (auto &param : nn->named_parameters())
                    {
                        if (param.key().find("weight") != std::string::npos)
                        {
                            reg = reg + lambda2 * param.value().square().sum()
                                      + lambda1 * param.value().abs().sum();
                        }
                    }
                }

                if (type == 1)
                {
                    // Targets are now (delta, s), not (n1, n2) -- see the
                    // residual parameterization in data_gen.h. The old deadzone
                    // mask keyed off "target normal has near-zero magnitude" to
                    // find absent planes; under this parameterization an absent
                    // plane gives |delta| ~ 0.5 or 1.5 and |s| ~ 1, so that test
                    // no longer identifies anything and would mask the wrong
                    // rows. Plain MSE is correct here: the reparameterization
                    // already puts the targets on a scale where a present plane
                    // and an absent one are both ordinary regression targets.
                    loss = critereon_MSE(check, comp);
                    //loss = permutation_invariant_normal_loss(check, comp);
                }
                else if (type == 2)
                {
                    loss = critereon_BCE(check, comp);
                    running_correct += ((check > 0.5) == comp).sum().item<long>();
                }
                else if (type == 3)
                {
                    auto target_classes = torch::argmax(comp, 1);
                    loss = critereon_CE(check, target_classes);
                    running_correct += torch::argmax(check, 1).eq(target_classes).sum().item<long>();
                }
                else
                {
                    loss = critereon_MSE(check, comp);
                }
                if (use_reg) loss = loss + reg;

                running_loss += (loss.detach().to(torch::kFloat64).reshape({1})
                                 * static_cast<double>(batch.data.size(0)));
                running_count += batch.data.size(0);

                // Set the LR for this step before stepping.
                double lr_now = lr_at_step(global_step, total_steps);
                for (auto &group : optimizer->param_groups())
                {
                    static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr_now);
                }

                optimizer->zero_grad();
                loss.backward();
                allreduce_gradients();
                optimizer->step();
                ++global_step;

                // ---------------------------------------------------------
                // Validation on a step cadence, not an epoch cadence, so the
                // evaluation frequency does not change when batch_size does.
                // Every rank hits this on the same global_step, so the
                // collectives inside stay matched.
                // ---------------------------------------------------------
                if (val_every_steps > 0 && global_step % val_every_steps == 0)
                {
                    double train_loss_local = running_loss.item<double>();
                    double train_count_local = static_cast<double>(running_count);

                    nn->eval();
                    double val_loss_local = 0.0;
                    double val_count_local = 0.0;
                    {
                        torch::NoGradGuard no_grad;
                        long vb = 0;
                        for (auto& vbatch : *data_loader_val)
                        {
                            if (vb >= val_steps) break;
                            ++vb;

                            auto vcheck = nn->forward(vbatch.data);
                            auto vcomp  = vbatch.target;
                            torch::Tensor vloss;
                            if (type == 1)
                            {
                                // Must match the training loss exactly, or the
                                // two curves are not comparable.
                                vloss = critereon_MSE(vcheck, vcomp);
                            }
                            else if (type == 2)
                            {
                                vloss = critereon_BCE(vcheck, vcomp);
                            }
                            else if (type == 3)
                            {
                                auto vt = torch::argmax(vcomp, 1);
                                vloss = critereon_CE(vcheck, vt);
                            }
                            else
                            {
                                vloss = critereon_MSE(vcheck, vcomp);
                            }
                            val_loss_local += vloss.item<double>() * vbatch.data.size(0);
                            val_count_local += vbatch.data.size(0);
                        }
                    }
                    nn->train();

                    // Reduce sums and counts separately so the reported means
                    // are correct even when ranks saw different sample counts.
                    double buf_in[4]  = {train_loss_local, train_count_local,
                                         val_loss_local, val_count_local};
                    double buf_out[4] = {0, 0, 0, 0};
                    if (numranks > 1)
                    {
                        MPI_Allreduce(buf_in, buf_out, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    }
                    else
                    {
                        for (int i = 0; i < 4; ++i) buf_out[i] = buf_in[i];
                    }
                    double train_mean = buf_out[1] > 0 ? buf_out[0] / buf_out[1] : 0.0;
                    double val_mean   = buf_out[3] > 0 ? buf_out[2] / buf_out[3] : 0.0;

                    if (rank == 0)
                    {
                        std::cout << global_step << " " << train_mean << " "
                                  << val_mean << " " << lr_now << std::endl;
                        std::cout.flush();
                    }

                    // Early stopping with patience, evaluated on every
                    // validation pass rather than every 100th epoch, and
                    // checkpointing the best model rather than the latest.
                    int stop_flag = 0;
                    if (val_mean < best_val)
                    {
                        best_val = val_mean;
                        evals_without_improvement = 0;
                        evals_since_plateau_drop = 0;
                        if (rank == 0)
                        {
                            torch::save(nn, out);
                        }
                    }
                    else
                    {
                        ++evals_without_improvement;

                        // Plateau LR decay fires on a shorter patience than
                        // early stopping, so the run gets several chances to
                        // shrink the step size and keep descending before it
                        // is abandoned. val_mean is identical on every rank,
                        // so plateau_lr_scale stays in sync without a collective.
                        if (use_plateau_decay)
                        {
                            ++evals_since_plateau_drop;
                            if (evals_since_plateau_drop >= plateau_patience)
                            {
                                plateau_lr_scale = std::max(plateau_lr_scale * plateau_factor,
                                                            plateau_min_lr_factor);
                                evals_since_plateau_drop = 0;
                                if (rank == 0)
                                {
                                    std::cout << "# lr drop at step " << global_step
                                              << " -> scale " << plateau_lr_scale << std::endl;
                                    std::cout.flush();
                                }
                            }
                        }

                        if (evals_without_improvement >= patience_evals)
                        {
                            stop_flag = 1;
                        }
                    }

                    // Broadcast the decision so all ranks leave together.
                    if (numranks > 1)
                    {
                        MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    }
                    if (stop_flag)
                    {
                        stop = true;
                        if (rank == 0)
                        {
                            std::cout << "# early stop at step " << global_step
                                      << ", best val " << best_val << std::endl;
                            std::cout.flush();
                        }
                        break;
                    }

                    running_loss.zero_();
                    running_count = 0;
                    running_correct = 0;
                }
            }

            if ((type == 2 || type == 3) && rank == 0 && running_count > 0)
            {
                std::cout << "# epoch " << epoch << " acc "
                          << static_cast<double>(running_correct) / running_count << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << "# done, best val " << best_val
                      << " after " << global_step << " steps" << std::endl;
            std::cout.flush();
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
                torch::NoGradGuard no_grad;

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
        torch::load(nn, in);
    }
}