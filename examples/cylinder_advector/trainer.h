// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2023 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.


#include <torch/torch.h>
#include "neural_network.h"
#include "irl/geometry/general/normal.h"

using namespace std;

class trainer
{
private:
    shared_ptr<model> nn;
    
public:
    trainer();
    void init();
    void load_model(string);
    IRL::Normal get_normal(vector<double>*);
};