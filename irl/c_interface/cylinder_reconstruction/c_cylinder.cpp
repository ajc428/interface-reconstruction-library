// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/c_interface/cylinder_reconstruction/c_cylinder.h"
#include "irl/geometry/general/unit_quaternion.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/helpers/mymath.h"
#include "irl/interface_reconstruction_methods/progressive_radius_solver_cylinder.h"

#include <Eigen/Dense>
#include <iostream>

extern "C" {

void c_Cylinder_new(c_Cylinder* a_self) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr == nullptr);
  a_self->is_owning = true;
  a_self->obj_ptr = new IRL::Cylinder;
  *a_self->obj_ptr = IRL::Cylinder();
}

void c_Cylinder_delete(c_Cylinder* a_self) {
  if (a_self->is_owning) {
    delete a_self->obj_ptr;
  }
  a_self->obj_ptr = nullptr;
  a_self->is_owning = false;
}

void c_Cylinder_setDatum(c_Cylinder* a_self, const double* a_datum) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setDatum(IRL::Pt::fromRawDoublePointer(a_datum));
}

void c_Cylinder_setReferenceFrame(c_Cylinder* a_self,
                                    const double* a_normal1,
                                    const double* a_normal2,
                                    const double* a_normal3) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setReferenceFrame(
      IRL::ReferenceFrame(IRL::Normal::fromRawDoublePointer(a_normal1),
                          IRL::Normal::fromRawDoublePointer(a_normal2),
                          IRL::Normal::fromRawDoublePointer(a_normal3)));
}

void c_Cylinder_setAlignedCylinder(c_Cylinder* a_self,
                                       const double* a_coeff_a,
                                       const double* a_coeff_b) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setAlignedCylinder(
      IRL::AlignedCylinder(std::array<double, 2>{(*a_coeff_a), (*a_coeff_b)}));
}

void c_Cylinder_setAlignedCylinderFlip(c_Cylinder* a_self,
                                       const double* a_coeff_a,
                                       const double* a_coeff_b,
                                       const double* a_coeff_f) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setAlignedCylinder(
      IRL::AlignedCylinder(std::array<double, 3>{(*a_coeff_a), (*a_coeff_b), (*a_coeff_f)}));
}

void c_Cylinder_copy(c_Cylinder* a_self,
                       const c_Cylinder* a_other_planar_separator) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(a_other_planar_separator != nullptr);
  assert(a_other_planar_separator->obj_ptr != nullptr);
  (*a_self->obj_ptr) = (*a_other_planar_separator->obj_ptr);
}

void c_Cylinder_getDatum(c_Cylinder* a_self, double* a_datum) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_datum[0] = (*a_self->obj_ptr).getDatum()[0];
  a_datum[1] = (*a_self->obj_ptr).getDatum()[1];
  a_datum[2] = (*a_self->obj_ptr).getDatum()[2];
}

void c_Cylinder_getReferenceFrame(c_Cylinder* a_self, double* a_frame) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_frame[0] = (*a_self->obj_ptr).getReferenceFrame()[0][0];
  a_frame[1] = (*a_self->obj_ptr).getReferenceFrame()[0][1];
  a_frame[2] = (*a_self->obj_ptr).getReferenceFrame()[0][2];
  a_frame[3] = (*a_self->obj_ptr).getReferenceFrame()[1][0];
  a_frame[4] = (*a_self->obj_ptr).getReferenceFrame()[1][1];
  a_frame[5] = (*a_self->obj_ptr).getReferenceFrame()[1][2];
  a_frame[6] = (*a_self->obj_ptr).getReferenceFrame()[2][0];
  a_frame[7] = (*a_self->obj_ptr).getReferenceFrame()[2][1];
  a_frame[8] = (*a_self->obj_ptr).getReferenceFrame()[2][2];
}

void c_Cylinder_getAlignedCylinder(c_Cylinder* a_self,
                                       double* a_aligned_cylinder) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_aligned_cylinder[0] = (*a_self->obj_ptr).getAlignedCylinder().r();
  a_aligned_cylinder[1] = (*a_self->obj_ptr).getAlignedCylinder().b();
  a_aligned_cylinder[2] = (*a_self->obj_ptr).getAlignedCylinder().f();
}

double c_Cylinder_getCurvature(c_Cylinder* a_self, c_RectCub* a_cell) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(a_cell != nullptr);
  assert(a_cell->obj_ptr != nullptr);
  IRL::RectangularCuboid cube = (*a_cell->obj_ptr);
  IRL::Cylinder cylinder = (*a_self->obj_ptr);
  auto moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<
      IRL::Volume, IRL::CylinderParametrizedSurfaceOutput>>(cube, cylinder);
  return moments.getSurface().getAverageMeanCurvature();
}

double c_Cylinder_getSurfaceArea(c_Cylinder* a_self, c_RectCub* a_cell) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(a_cell != nullptr);
  assert(a_cell->obj_ptr != nullptr);
  IRL::RectangularCuboid cube = (*a_cell->obj_ptr);
  IRL::Cylinder cylinder = (*a_self->obj_ptr);
  auto moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<
      IRL::Volume, IRL::CylinderParametrizedSurfaceOutput>>(cube, cylinder);
  return moments.getSurface().getSurfaceArea();
}

void c_Cylinder_printToScreen(const c_Cylinder* a_self) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  std::cout << (*a_self->obj_ptr);
}

}  // end extern C
