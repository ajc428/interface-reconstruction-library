// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_INTERFACE_RECONSTRUCTION_METHODS_CYLINDER_NEIGHBORHOOD_H_
#define IRL_INTERFACE_RECONSTRUCTION_METHODS_CYLINDER_NEIGHBORHOOD_H_

#include <float.h>

#include <cassert>
#include <string>

#include "irl/generic_cutting/cut_polygon.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/moments/cell_collection.h"
#include "irl/moments/cell_grouped_moments.h"
#include "irl/parameters/defined_types.h"

namespace IRL {

class cylinderNeighborhood {
  using CGD = CellGroupedMoments<RectangularCuboid, VolumeMoments>;

 public:
  /// \brief Default constructor.
  cylinderNeighborhood(void) = default;

  /// \brief Construct a CellGroupedMoments and add it
  /// to the collection for index i,j,k.
  void setMember(const RectangularCuboid* a_rectangular_cuboid,
                 const VolumeMoments* a_volume_moments, const int i,
                 const int j, const int k);

  /// \brief Return the cell stored at the index i,j,k
  const RectangularCuboid& getCell(const int i, const int j,
                                   const int k) const;

  /// \brief Return moments stored at the index i,j,k
  VolumeMoments getStoredMoments(const int i, const int j, const int k) const;

  /// \brief Set size of the neighborhood.
  void resize(const UnsignedIndex_t a_size);

  /// \brief Default destructor.
  ~cylinderNeighborhood(void) = default;

 private:
  /// \brief Calculate linear index from i,j,k.
  UnsignedIndex_t calculateLinearIndex(const int i, const int j,
                                       const int k) const;

  CellCollection<CGD>
      collection_m;  ///< \brief Collection that holds correct moments.
};

}  // namespace IRL

#endif // IRL_INTERFACE_RECONSTRUCTION_METHODS_CYLINDER_NEIGHBORHOOD_H_
