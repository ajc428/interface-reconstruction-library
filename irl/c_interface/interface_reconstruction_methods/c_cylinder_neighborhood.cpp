// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/c_interface/interface_reconstruction_methods/c_cylinder_neighborhood.h"

#include <cassert>

extern "C" {

void c_cylinderNeigh_new(c_cylinderNeigh* a_self) {
  assert(a_self->obj_ptr == nullptr);
  a_self->obj_ptr = new IRL::cylinderNeighborhood;
}

void c_cylinderNeigh_delete(c_cylinderNeigh* a_self) {
  delete a_self->obj_ptr;
  a_self->obj_ptr = nullptr;
}

void c_cylinderNeigh_setSize(c_cylinderNeigh* a_self, const int* a_size) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->resize(static_cast<IRL::UnsignedIndex_t>(*a_size));
}

void c_cylinderNeigh_setMember(c_cylinderNeigh* a_self,
                             const c_RectCub* a_rectangular_cuboid,
                             const c_VM* a_volume_moments,
                             const int* i, const int* j, const int* k) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(a_rectangular_cuboid != nullptr);
  assert(a_rectangular_cuboid->obj_ptr != nullptr);
  assert(a_volume_moments != nullptr);
  assert(a_volume_moments->obj_ptr != nullptr);
  a_self->obj_ptr->setMember(a_rectangular_cuboid->obj_ptr,
                             a_volume_moments->obj_ptr, *i, *j, *k);
}
}
