// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "irl/interface_reconstruction_methods/cylinder_neighborhood.h"

namespace IRL {

/// \brief Construct a CellGroupedMoments and add it
/// to the collection for index i,j,k.
void cylinderNeighborhood::setMember(
    const RectangularCuboid* a_rectangular_cuboid,
    const VolumeMoments* a_volume_moments, const int i, const int j,
    const int k) {
  assert(a_rectangular_cuboid != nullptr);
  assert(a_volume_moments != nullptr);
  collection_m[this->calculateLinearIndex(i, j, k)] =
      CGD(a_rectangular_cuboid, a_volume_moments);
}

/// \brief Return the cell stored at the index i,j,k
const RectangularCuboid& cylinderNeighborhood::getCell(const int i, const int j,
                                                     const int k) const {
  return collection_m.getCell(this->calculateLinearIndex(i, j, k));
}

/// \brief Return moments stored at the index i,j,k
VolumeMoments cylinderNeighborhood::getStoredMoments(const int i, const int j,
                                            const int k) const {
  return collection_m.getStoredMoments(this->calculateLinearIndex(i, j, k));
}

/// \brief Set size of the neighborhood.
void cylinderNeighborhood::resize(const UnsignedIndex_t a_size) {
  collection_m.resize(a_size);
}

/// \brief Calculate linear index from i,j,k.
UnsignedIndex_t cylinderNeighborhood::calculateLinearIndex(const int i,
                                                         const int j,
                                                         const int k) const {
  return static_cast<UnsignedIndex_t>((i + 2) + (j + 2) * 5 + (k + 2) * 25);
}

}  // namespace IRL
