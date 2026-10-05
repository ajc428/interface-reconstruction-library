// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2026 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Uniform periodic Cartesian grid on a box, and cell-centred fields on it with
// ghost layers. Interior cells are 0..n-1 in each direction, ghost cells
// -ghosts..-1 and n..n+ghosts-1.

#ifndef EXAMPLES_R2P_ADVECTOR_GRID_H_
#define EXAMPLES_R2P_ADVECTOR_GRID_H_

#include <cstddef>
#include <vector>

#include "irl/geometry/general/pt.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"

struct Grid {
  int n[3];          // interior cells per direction
  int ghosts;        // ghost layers on every side
  double lower[3];   // lower corner of the domain
  double h[3];       // cell size per direction

  Grid(const int cells, const int ghost_layers, const IRL::Pt& lower_corner, const IRL::Pt& upper_corner)
      : n{cells, cells, cells}, ghosts(ghost_layers) {
    for (int d = 0; d < 3; ++d) {
      lower[d] = lower_corner[d];
      h[d] = (upper_corner[d] - lower_corner[d]) / static_cast<double>(cells);
    }
  }

  // Face coordinate c along direction d (cell c spans [face(d,c), face(d,c+1)])
  // and cell centre; x/y/z and xm/ym/zm are the same per direction.
  double face(const int d, const int c) const { return lower[d] + static_cast<double>(c) * h[d]; }
  double centre(const int d, const int c) const { return 0.5 * (face(d, c) + face(d, c + 1)); }
  double x(const int i) const { return face(0, i); }
  double y(const int j) const { return face(1, j); }
  double z(const int k) const { return face(2, k); }
  double xm(const int i) const { return centre(0, i); }
  double ym(const int j) const { return centre(1, j); }
  double zm(const int k) const { return centre(2, k); }
  double length(const int d) const { return h[d] * static_cast<double>(n[d]); }

  IRL::RectangularCuboid cell(const int i, const int j, const int k) const {
    return IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(x(i), y(j), z(k)),
                                                   IRL::Pt(x(i + 1), y(j + 1), z(k + 1)));
  }
  IRL::Pt cellCentre(const int i, const int j, const int k) const { return IRL::Pt(xm(i), ym(j), zm(k)); }

  // Storage size and linear index (k fastest), ghosts included.
  std::size_t size() const {
    return static_cast<std::size_t>(n[0] + 2 * ghosts) * (n[1] + 2 * ghosts) * (n[2] + 2 * ghosts);
  }
  std::size_t index(const int i, const int j, const int k) const {
    return (static_cast<std::size_t>(i + ghosts) * (n[1] + 2 * ghosts) + (j + ghosts)) * (n[2] + 2 * ghosts) +
           (k + ghosts);
  }
  bool isGhost(const int i, const int j, const int k) const {
    return i < 0 || j < 0 || k < 0 || i >= n[0] || j >= n[1] || k >= n[2];
  }
};

// Loops over interior cells, or over all cells including ghosts.
#define FOR_INTERIOR(grid, i, j, k)                   \
  for (int i = 0; i < (grid).n[0]; ++i)               \
    for (int j = 0; j < (grid).n[1]; ++j)             \
      for (int k = 0; k < (grid).n[2]; ++k)
#define FOR_ALL(grid, i, j, k)                                        \
  for (int i = -(grid).ghosts; i < (grid).n[0] + (grid).ghosts; ++i)  \
    for (int j = -(grid).ghosts; j < (grid).n[1] + (grid).ghosts; ++j) \
      for (int k = -(grid).ghosts; k < (grid).n[2] + (grid).ghosts; ++k)

template <class T>
class Field {
 public:
  explicit Field(const Grid& grid) : grid_(&grid), values_(grid.size()) {}

  T& operator()(const int i, const int j, const int k) { return values_[grid_->index(i, j, k)]; }
  const T& operator()(const int i, const int j, const int k) const { return values_[grid_->index(i, j, k)]; }
  const Grid& grid() const { return *grid_; }

  // Periodic ghost cells: copies of the interior cells one period away.
  void fillGhosts() {
    const Grid& g = *grid_;
    FOR_ALL(g, i, j, k) {
      if (!g.isGhost(i, j, k)) continue;
      (*this)(i, j, k) = (*this)(wrap(i, g.n[0]), wrap(j, g.n[1]), wrap(k, g.n[2]));
    }
  }

  // Trilinear interpolation between cell centres.
  T interpolate(const IRL::Pt& p) const {
    const Grid& g = *grid_;
    int c[3];
    double w[3];
    for (int d = 0; d < 3; ++d) {
      // Cell holding p (truncated toward zero), kept so that c and c+1 exist.
      c[d] = static_cast<int>((p[d] - g.lower[d]) / g.h[d]);
      if (c[d] < -g.ghosts) c[d] = -g.ghosts;
      if (c[d] > g.n[d] + g.ghosts - 2) c[d] = g.n[d] + g.ghosts - 2;
      w[d] = (p[d] - g.centre(d, c[d])) / (g.centre(d, c[d] + 1) - g.centre(d, c[d]));
    }
    const Field& f = *this;
    const int i = c[0], j = c[1], k = c[2];
    return w[2] * (w[1] * (w[0] * f(i + 1, j + 1, k + 1) + (1.0 - w[0]) * f(i, j + 1, k + 1)) +
                   (1.0 - w[1]) * (w[0] * f(i + 1, j, k + 1) + (1.0 - w[0]) * f(i, j, k + 1))) +
           (1.0 - w[2]) * (w[1] * (w[0] * f(i + 1, j + 1, k) + (1.0 - w[0]) * f(i, j + 1, k)) +
                           (1.0 - w[1]) * (w[0] * f(i + 1, j, k) + (1.0 - w[0]) * f(i, j, k)));
  }

 private:
  static int wrap(const int i, const int n) { return ((i % n) + n) % n; }
  const Grid* grid_;
  std::vector<T> values_;
};

#endif  // EXAMPLES_R2P_ADVECTOR_GRID_H_
