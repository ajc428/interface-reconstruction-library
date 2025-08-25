// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_C_INTERFACE_CYLINDER_RECONSTRUCTION_C_CYLINDER_H_
#define IRL_C_INTERFACE_CYLINDER_RECONSTRUCTION_C_CYLINDER_H_

#include "irl/c_interface/geometry/polygons/c_polygon.h"
#include "irl/c_interface/geometry/polyhedrons/c_rectangular_cuboid.h"
#include "irl/c_interface/planar_reconstruction/c_separators.h"
#include "irl/data_structures/object_allocation_server.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/cylinder_reconstruction/cylinder.h"
#include "irl/planar_reconstruction/planar_separator.h"

extern "C" {

struct c_Cylinder {
  IRL::Cylinder* obj_ptr = nullptr;
  bool is_owning = false;
};

void c_Cylinder_new(c_Cylinder* a_self);

void c_Cylinder_delete(c_Cylinder* a_self);

void c_Cylinder_setDatum(c_Cylinder* a_self, const double* a_datum);

void c_Cylinder_setReferenceFrame(c_Cylinder* a_self,
                                    const double* a_normal1,
                                    const double* a_normal2,
                                    const double* a_normal3);

void c_Cylinder_setAlignedCylinder(c_Cylinder* a_self,
                                       const double* a_coeff_a,
                                       const double* a_coeff_b);

void c_Cylinder_copy(c_Cylinder* a_self,
                       const c_Cylinder* a_other_planar_separator);

void c_Cylinder_getDatum(c_Cylinder* a_self, double* a_datum);

void c_Cylinder_getReferenceFrame(c_Cylinder* a_self, double* a_frame);

void c_Cylinder_getAlignedCylinder(c_Cylinder* a_self,
                                       double* a_aligned_cylinder);

double c_Cylinder_getCurvature(c_Cylinder* a_self, c_RectCub* a_cell);

double c_Cylinder_getSurfaceArea(c_Cylinder* a_self, c_RectCub* a_cell);

void c_Cylinder_printToScreen(const c_Cylinder* a_self);

}  // end extern C

#endif  // IRL_C_INTERFACE_CYLINDER_RECONSTRUCTION_C_CYLINDER_H_
