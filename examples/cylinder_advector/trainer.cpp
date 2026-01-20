// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "examples/cylinder_advector/trainer.h"

trainer::trainer()
{
    init();
}

void trainer::init()
{
    nn = make_shared<model>(189,3,3,100,0);
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