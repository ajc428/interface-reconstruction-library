// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_C_INTERFACE_INTERFACE_RECONSTRUCTION_METHODS_C_CYLINDER_NEIGHBORHOOD_H_
#define IRL_C_INTERFACE_INTERFACE_RECONSTRUCTION_METHODS_C_CYLINDER_NEIGHBORHOOD_H_

#include "irl/c_interface/geometry/polyhedrons/c_rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/cylinder_neighborhood.h"
#include "irl/c_interface/moments/c_volume_moments.h"

extern "C" {

struct c_cylinderNeigh {
  IRL::cylinderNeighborhood* obj_ptr = nullptr;
};

void c_cylinderNeigh_new(c_cylinderNeigh* a_self);
void c_cylinderNeigh_delete(c_cylinderNeigh* a_self);
void c_cylinderNeigh_setSize(c_cylinderNeigh* a_self, const int* a_size);
void c_cylinderNeigh_setMember(c_cylinderNeigh* a_self,
                             const c_RectCub* a_rectangular_cuboid,
                             const c_VM* a_volume_moments,
                             const int* i, const int* j, const int* k);
}

#endif // IRL_C_INTERFACE_INTERFACE_RECONSTRUCTION_METHODS_C_CYLINDER_NEIGHBORHOOD_H_
