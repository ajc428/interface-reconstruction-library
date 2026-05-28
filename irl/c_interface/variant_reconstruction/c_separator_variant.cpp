// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2019 Robert Chiodi <robert.chiodi@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/c_interface/variant_reconstruction/c_separator_variant.h"

#include <iostream>

#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/parameters/constants.h"

extern "C" {

void c_SeparatorVariant_new(c_SeparatorVariant* a_self) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr == nullptr);
  a_self->is_owning = true;
  a_self->obj_ptr = new IRL::SeparatorVariant;
}

void c_SeparatorVariant_newFromObjectAllocationServer(
    c_SeparatorVariant* a_self,
    c_ObjServer_SeparatorVariant* a_object_allocation_server) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr == nullptr);
  assert(a_object_allocation_server != nullptr);
  assert(a_object_allocation_server->obj_ptr != nullptr);
  a_self->is_owning = false;
  a_self->obj_ptr = a_object_allocation_server->obj_ptr->getNewObject();
}

void c_SeparatorVariant_delete(c_SeparatorVariant* a_self) {
  if (a_self->is_owning) {
    delete a_self->obj_ptr;
  }
  a_self->obj_ptr = nullptr;
  a_self->is_owning = false;
}

void c_SeparatorVariant_setNumberOfPlanes(c_SeparatorVariant* a_self,
                                          const int* a_number_to_set) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(*a_number_to_set >= 0);
  a_self->obj_ptr->setToPlanarSeparator();
  if (IRL::PlanarSeparator* separator =
          std::get_if<IRL::PlanarSeparator>(a_self->obj_ptr)) {
    separator->setNumberOfPlanes(
        static_cast<IRL::UnsignedIndex_t>(*a_number_to_set));
  }
}

void c_SeparatorVariant_setPlane(c_SeparatorVariant* a_self,
                                 const int* a_plane_index_to_set,
                                 const double* a_normal,
                                 const double* a_distance) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(*a_plane_index_to_set >= 0);
  a_self->obj_ptr->setToPlanarSeparator();
  if (IRL::PlanarSeparator* separator =
          std::get_if<IRL::PlanarSeparator>(a_self->obj_ptr)) {
    (*separator)[static_cast<IRL::UnsignedIndex_t>(*a_plane_index_to_set)] =
        IRL::Plane(IRL::Normal::fromRawDoublePointer(a_normal), *a_distance);
  }
}

void c_SeparatorVariant_setAlignedCylinder(c_SeparatorVariant* a_self,
                                 const double* b,
                                 const double* r) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setToCylinder();
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    separator->setAlignedCylinder(IRL::AlignedCylinder(std::array<double,2>{*b,*r}));
  }
}

void c_SeparatorVariant_setAlignedCylinder_flip(c_SeparatorVariant* a_self,
                                 const double* b,
                                 const double* r,
                                 const double* f) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setToCylinder();
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    separator->setAlignedCylinder(IRL::AlignedCylinder(std::array<double,3>{*b,*r,*f}));
  }
}

void c_SeparatorVariant_setDatum(c_SeparatorVariant* a_self, const double* a_datum) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setToCylinder();
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    separator->setDatum(IRL::Pt::fromRawDoublePointer(a_datum));
  }
}

void c_SeparatorVariant_setReferenceFrame(c_SeparatorVariant* a_self,
                                    const double* a_normal1,
                                    const double* a_normal2,
                                    const double* a_normal3) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  a_self->obj_ptr->setToCylinder();
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    separator->setReferenceFrame(
      IRL::ReferenceFrame(IRL::Normal::fromRawDoublePointer(a_normal1),
                          IRL::Normal::fromRawDoublePointer(a_normal2),
                          IRL::Normal::fromRawDoublePointer(a_normal3)));
  }
}

void c_SeparatorVariant_copy(
    c_SeparatorVariant* a_self,
    const c_SeparatorVariant* a_other_planar_separator) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(a_other_planar_separator != nullptr);
  assert(a_other_planar_separator->obj_ptr != nullptr);
  (*a_self->obj_ptr) = (*a_other_planar_separator->obj_ptr);
}

int c_SeparatorVariant_getNumberOfPlanes(const c_SeparatorVariant* a_self) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  if (IRL::PlanarSeparator* separator =
          std::get_if<IRL::PlanarSeparator>(a_self->obj_ptr)) {
    return static_cast<int>(separator->getNumberOfPlanes());
  } else {
    return 0;
  }
}

void c_SeparatorVariant_getPlane(c_SeparatorVariant* a_self, const int* a_index,
                                 double* a_plane_listed) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(*a_index >= 0);
  if (IRL::PlanarSeparator* separator =
          std::get_if<IRL::PlanarSeparator>(a_self->obj_ptr)) {
    assert(static_cast<IRL::UnsignedIndex_t>(*a_index) <
         separator->getNumberOfPlanes());
    a_plane_listed[0] =
        (*separator)[static_cast<IRL::UnsignedIndex_t>(*a_index)].normal()[0];
    a_plane_listed[1] =
        (*separator)[static_cast<IRL::UnsignedIndex_t>(*a_index)].normal()[1];
    a_plane_listed[2] =
        (*separator)[static_cast<IRL::UnsignedIndex_t>(*a_index)].normal()[2];
    a_plane_listed[3] =
        (*separator)[static_cast<IRL::UnsignedIndex_t>(*a_index)].distance();
  }
}

void c_SeparatorVariant_getAlignedCylinder(c_SeparatorVariant* a_self,
                                       double* a_aligned_cylinder) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    a_aligned_cylinder[0] = separator->getAlignedCylinder().r();
    a_aligned_cylinder[1] = separator->getAlignedCylinder().b();
    a_aligned_cylinder[2] = separator->getAlignedCylinder().f();
  }
}

void c_SeparatorVariant_getDatum(c_SeparatorVariant* a_self, double* a_datum) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    a_datum[0] = separator->getDatum()[0];
    a_datum[1] = separator->getDatum()[1];
    a_datum[2] = separator->getDatum()[2];
  }
}

void c_SeparatorVariant_getReferenceFrame(c_SeparatorVariant* a_self, double* a_frame) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    a_frame[0] = separator->getReferenceFrame()[0][0];
    a_frame[1] = separator->getReferenceFrame()[0][1];
    a_frame[2] = separator->getReferenceFrame()[0][2];
    a_frame[3] = separator->getReferenceFrame()[1][0];
    a_frame[4] = separator->getReferenceFrame()[1][1];
    a_frame[5] = separator->getReferenceFrame()[1][2];
    a_frame[6] = separator->getReferenceFrame()[2][0];
    a_frame[7] = separator->getReferenceFrame()[2][1];
    a_frame[8] = separator->getReferenceFrame()[2][2];
  }
}

double c_SeparatorVariant_getSurfaceArea(c_SeparatorVariant* a_self, c_RectCub* a_cell) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(a_cell != nullptr);
  assert(a_cell->obj_ptr != nullptr);
  double area = 0.0;
  if (IRL::Cylinder* separator =
          std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    IRL::RectangularCuboid cube = (*a_cell->obj_ptr);
    IRL::Cylinder cylinder = (*separator);
    auto moments = IRL::getVolumeMoments<IRL::AddSurfaceOutput<
        IRL::Volume, IRL::CylinderParametrizedSurfaceOutput>>(cube, cylinder);
    area = moments.getSurface().getSurfaceArea();
  }
  return area;
}

bool c_SeparatorVariant_isFlipped(const c_SeparatorVariant* a_self) {
  assert(a_self != nullptr);
  if (IRL::PlanarSeparator* separator =
          std::get_if<IRL::PlanarSeparator>(a_self->obj_ptr)) {
    return separator->isFlipped();
  } else if (IRL::Paraboloid* paraboloid =
                 std::get_if<IRL::Paraboloid>(a_self->obj_ptr)) {
    return paraboloid->isFlipped();
  } else if (IRL::Cylinder* cylinder =
                 std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    return cylinder->isFlipped();
  } else {
    throw std::runtime_error("Variant type unknown");
  }
}

void c_SeparatorVariant_printToScreen(const c_SeparatorVariant* a_self) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  if (IRL::PlanarSeparator* separator =
          std::get_if<IRL::PlanarSeparator>(a_self->obj_ptr)) {
    std::cout << (*separator);
  } else if (IRL::Paraboloid* paraboloid =
                 std::get_if<IRL::Paraboloid>(a_self->obj_ptr)) {
    std::cout << (*paraboloid);
  } else if (IRL::Cylinder* cylinder =
                 std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    std::cout << (*cylinder);
  } else {
    throw std::runtime_error("Variant type unknown");
  }
}

void c_SeparatorVariant_shift(c_SeparatorVariant* a_self,
                              const double* a_shift) {
  assert(a_self != nullptr);
  assert(a_self->obj_ptr != nullptr);
  assert(a_shift != nullptr);
  const IRL::Pt shift = IRL::Pt::fromRawDoublePointer(a_shift);
  if (IRL::PlanarSeparator* separator =
          std::get_if<IRL::PlanarSeparator>(a_self->obj_ptr)) {
    for (auto& plane : *separator) {
      plane.distance() += plane.normal() * shift;
    }
  } else if (IRL::Paraboloid* paraboloid =
                 std::get_if<IRL::Paraboloid>(a_self->obj_ptr)) {
    const IRL::Pt& datum = paraboloid->getDatum();
    paraboloid->setDatum(datum + shift);
  } else if (IRL::Cylinder* cylinder =
                 std::get_if<IRL::Cylinder>(a_self->obj_ptr)) {
    const IRL::Pt& datum = cylinder->getDatum();
    cylinder->setDatum(datum + shift);
  } else {
    throw std::runtime_error("Variant type unknown");
  }
}

}  // end extern C
