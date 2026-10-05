// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2026 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// R2P-Net interface reconstruction, in one file.
//
// From each cell's liquid volume fraction and liquid/gas centroids, builds one
// plane (a resolved interface) or two (a film thinner than the cell). Per mixed
// cell:
//
//  1. Film phase. The film is the phase between the two faces: of liquid and
//     gas, the one whose 3^3 centroid cloud is flatter.
//  2. Routing to R2P-Net (two planes possible) or PLICNet (one plane):
//     - the ML classifier on the 5^3 stencil: sheets and sheet ends;
//     - the thin-film guard: the other phase lies on both sides of the film in
//       separate regions, so the film continues through the cell;
//     - a very thin film (thinner than kPcaSlabThickness over the 3^3 stencil).
//  3. PLICNet: one normal; the distance matches the volume fraction.
//  4. R2P-Net: two face normals from the canonicalised 3^3 moments.
//     - Very thin film: both normals become +- the stencil's PCA direction.
//     - One predicted normal collapsed (|n| < kTwoPlaneMagnitude) but the guard
//       holds: a parallel slab around the other normal.
//     - Two planes: Newton places them to match the volume fraction and the
//       film's centroid; pinch prevention then opens converging planes that
//       would pinch the film off inside the cell, unless the film ends there
//       (topological edge sensor) or is thick.
//     - One plane: the network's normal or PLICNet's, whichever matches the
//       cell's centroids better.
//  5. Pass 2, every two-plane cell: fit both faces jointly to the 3^3 stencil's
//     interface polygons, rotate the normals toward the fit (limited), re-place
//     the planes, and keep one plane instead if it matches the film centroid
//     about as well (never in a guarded or very thin film).
//
// Networks (generated, in networks/): r2pnet.h (192 -> 256 x 3 -> 6),
// plicnet.h (189 -> 100 x 3 -> 3) and ml_classifier.h (500 -> 256 -> 64 -> 32
// -> 6), each with its input canonicalisation.

#ifndef EXAMPLES_R2P_ADVECTOR_R2P_NET_H_
#define EXAMPLES_R2P_ADVECTOR_R2P_NET_H_

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "irl/generic_cutting/cut_polygon.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/machine_learning_reconstruction/plic_paraboloid.h"
#include "irl/moments/separated_volume_moments.h"
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/r2p_advector/grid.h"
#include "examples/r2p_advector/networks/ml_classifier.h"
#include "examples/r2p_advector/networks/plicnet.h"
#include "examples/r2p_advector/networks/r2pnet.h"

namespace r2p {

// ===========================================================================
// Parameters
// ===========================================================================

constexpr double kTwoPlaneMagnitude = 0.85;   // both network normals at least this long: two planes
constexpr double kPcaSlabThickness = 0.005;   // cells; thinnest films in the training data
constexpr double kPinchGapFraction = 0.1;     // thinnest film in the cell / mean thickness, at least
constexpr double kThickFilmFraction = 0.01;   // edge sensor: film VF that makes a blocked side "thick"
constexpr double kGuardThinVolume = 1.0;      // guard: film volume in the 3^3 block below which ...
constexpr int kGuardNearDistance2 = 5;        // ... "near the cell" extends to squared distance 5
constexpr int kSheetClass = 4, kSheetEndClass = 6;   // ML classifier ids routed to R2P-Net

// Pass 2.
constexpr double kMaxRotation = 0.35;         // rad, per normal
constexpr double kMaxSplayChange = 0.30;      // rad, change of the angle between the faces
constexpr double kMinFaceAreaFraction = 0.10; // each face's share of the stencil's polygon area
constexpr int kMinPolygonsPerFace = 6;
constexpr double kFitRadius = 2.5;            // cells, weight kernel radius
constexpr double kSplayPenalty = 0.015;       // ridge on the faces' differing slopes
constexpr double kCurvaturePenalty = 1.0e-3;  // ridge on the shared curvature
constexpr double kMaxFitResidual = 0.25;      // cells, RMS
constexpr double kMinBisector = 1.0e-3;
constexpr double kPlaneDropBias = 1.05;       // one plane wins if its centroid error <= this x two planes'

using IRL::global_constants::VF_HIGH;
using IRL::global_constants::VF_LOW;

// ===========================================================================
// Diagnostics (optional output, one value per cell)
// ===========================================================================

enum Route { kPure = -1, kPlicNet = 0, kR2PNet = 1 };
enum Slab { kNoSlab = 0, kGuardSlab = 2, kPcaSlab = 3 };   // 1: the retired thin-film snap
enum Edge { kNotComputed = -1, kContinuing = 0, kEdge = 1, kThick = 2 };

struct Diagnostics {
  explicit Diagnostics(const Grid& grid)
      : route(grid), classifier(grid), guard(grid), slab(grid), edge(grid), unpinch(grid) {}
  Field<int> route;       // Route
  Field<int> classifier;  // ML classifier id (0: not classified)
  Field<int> guard;       // 1: the guard sent the cell to R2P-Net, 2: it held for an R2P-Net cell
  Field<int> slab;        // Slab
  Field<int> edge;        // Edge
  Field<double> unpinch;  // pinch prevention's rotation toward the mean normal (0: none, 1: slab)
};

// ===========================================================================
// Dense networks, evaluated one cell at a time
// ===========================================================================

// y = b + W x for one layer, with W stored transposed (row i: input i's weights
// to every output). Each output sums its inputs in increasing order, as the
// generated loops do, so results are bit-identical to them; zero inputs (most
// of a sparse stencil) are skipped and the output loop vectorises.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
__attribute__((target_clones("avx2", "default")))
#endif
inline void denseLayer(const double* __restrict weights_t, const double* __restrict bias, const int n_in,
                       const int n_out, const double* __restrict x, double* __restrict y, const bool relu) {
  for (int o = 0; o < n_out; ++o) y[o] = bias[o];
  for (int i = 0; i < n_in; ++i) {
    if (x[i] == 0.0) continue;
    const double* __restrict w = weights_t + static_cast<long>(i) * n_out;
    for (int o = 0; o < n_out; ++o) y[o] += x[i] * w[o];
  }
  if (relu)
    for (int o = 0; o < n_out; ++o) y[o] = std::max(y[o], 0.0);
}

// Multilayer perceptron: ReLU on every layer but the last.
class DenseNetwork {
 public:
  template <int Out, int In>
  DenseNetwork& addLayer(const double (&weights)[Out][In], const double (&bias)[Out]) {
    Layer layer{In, Out, std::vector<double>(static_cast<std::size_t>(In) * Out), {bias, bias + Out}};
    for (int o = 0; o < Out; ++o)
      for (int i = 0; i < In; ++i) layer.weights_t[static_cast<std::size_t>(i) * Out + o] = weights[o][i];
    layers_.push_back(std::move(layer));
    return *this;
  }
  void evaluate(const double* input, double* output) const {
    std::array<double, kMaxWidth> buffer[2];
    const double* x = input;
    for (std::size_t l = 0; l < layers_.size(); ++l) {
      const bool last = l + 1 == layers_.size();
      double* y = last ? output : buffer[l % 2].data();
      denseLayer(layers_[l].weights_t.data(), layers_[l].bias.data(), layers_[l].n_in, layers_[l].n_out, x, y,
                 !last);
      x = y;
    }
  }

 private:
  static constexpr int kMaxWidth = 512;
  struct Layer {
    int n_in, n_out;
    std::vector<double> weights_t, bias;
  };
  std::vector<Layer> layers_;
};

inline const DenseNetwork& classifierNetwork() {
  namespace w = ml_classifier::detail;
  static const DenseNetwork network = DenseNetwork()
                                          .addLayer(w::lay1_weight, w::lay1_bias)
                                          .addLayer(w::lay2_weight, w::lay2_bias)
                                          .addLayer(w::lay3_weight, w::lay3_bias)
                                          .addLayer(w::lay4_weight, w::lay4_bias);
  return network;
}
inline const DenseNetwork& plicNetwork() {
  static const DenseNetwork network = DenseNetwork()
                                          .addLayer(plicnet::lay1_weight, plicnet::lay1_bias)
                                          .addLayer(plicnet::lay2_weight, plicnet::lay2_bias)
                                          .addLayer(plicnet::lay3_weight, plicnet::lay3_bias)
                                          .addLayer(plicnet::lay4_weight, plicnet::lay4_bias);
  return network;
}
inline const DenseNetwork& r2pNetwork() {
  static const DenseNetwork network = DenseNetwork()
                                          .addLayer(r2pnet::lay1_weight, r2pnet::lay1_bias)
                                          .addLayer(r2pnet::lay2_weight, r2pnet::lay2_bias)
                                          .addLayer(r2pnet::lay3_weight, r2pnet::lay3_bias)
                                          .addLayer(r2pnet::lay4_weight, r2pnet::lay4_bias);
  return network;
}

// ===========================================================================
// The flow fields, and one phase of them
// ===========================================================================

struct Fields {
  const Grid& grid;
  const Field<double>& vf;   // liquid volume fraction
  const Field<IRL::Pt>& liquid_centroid;
  const Field<IRL::Pt>& gas_centroid;
};

// Liquid or gas: its volume fraction and centroid in any cell.
struct Phase {
  const Fields& fields;
  bool is_gas;
  double fraction(const int i, const int j, const int k) const {
    return is_gas ? 1.0 - fields.vf(i, j, k) : fields.vf(i, j, k);
  }
  const IRL::Pt& centroid(const int i, const int j, const int k) const {
    return is_gas ? fields.gas_centroid(i, j, k) : fields.liquid_centroid(i, j, k);
  }
  const IRL::Pt& otherCentroid(const int i, const int j, const int k) const {
    return is_gas ? fields.liquid_centroid(i, j, k) : fields.gas_centroid(i, j, k);
  }
};

// Grid-index frame (unit cells, as the networks see them) to physical normal.
inline IRL::Normal toPhysical(IRL::Normal n, const Grid& grid) {
  for (int d = 0; d < 3; ++d) n[d] *= grid.h[d];
  n.normalize();
  return n;
}

// ===========================================================================
// Stencil geometry: centroid clouds and film thickness
// ===========================================================================

// Centroids of the phase in the 3^3 cells that hold it (fraction > VF_LOW).
inline int centroidCloud(const Phase& phase, const int i, const int j, const int k, Eigen::Vector3d* points) {
  int count = 0;
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) {
        if (!(phase.fraction(ii, jj, kk) > VF_LOW)) continue;
        const IRL::Pt& c = phase.centroid(ii, jj, kk);
        points[count++] = Eigen::Vector3d(c[0], c[1], c[2]);
      }
  return count;
}

// Eigen-decomposition of a point cloud's covariance (eigenvalues ascending).
inline Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> cloudShape(const Eigen::Vector3d* points, const int count) {
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (int q = 0; q < count; ++q) mean += points[q];
  mean = mean / double(count);
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (int q = 0; q < count; ++q) covariance += (points[q] - mean) * (points[q] - mean).transpose();
  return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(covariance);
}

// Film phase: the phase whose 3^3 centroid cloud is flatter (smaller ratio of
// smallest to largest covariance eigenvalue). With fewer than 3 cells of either
// phase, the minority phase of the 3^3 block.
inline bool filmIsGas(const Fields& fields, const int i, const int j, const int k) {
  auto flatness = [&](const bool gas) {
    Eigen::Vector3d points[27];
    const int count = centroidCloud(Phase{fields, gas}, i, j, k, points);
    if (count < 3) return -1.0;
    const Eigen::Vector3d eigenvalues = cloudShape(points, count).eigenvalues();
    return eigenvalues(2) > 1.0e-30 ? std::max(0.0, eigenvalues(0)) / eigenvalues(2) : -1.0;
  };
  const double liquid = flatness(false), gas = flatness(true);
  if (liquid < 0.0 || gas < 0.0) {
    double vf_sum = 0.0;
    for (int ii = i - 1; ii <= i + 1; ++ii)
      for (int jj = j - 1; jj <= j + 1; ++jj)
        for (int kk = k - 1; kk <= k + 1; ++kk) vf_sum += fields.vf(ii, jj, kk);
    return vf_sum >= 0.5 * 27.0;
  }
  return gas < liquid;
}

// Normal of the film phase's 3^3 centroid cloud (its thinnest direction); on a
// thin film the centroids lie on its mid-surface. False with fewer than 6 cells.
inline bool filmPcaNormal(const Phase& film, const int i, const int j, const int k, IRL::Normal* normal) {
  Eigen::Vector3d points[27];
  const int count = centroidCloud(film, i, j, k, points);
  if (count < 6) return false;
  const Eigen::Vector3d n = cloudShape(points, count).eigenvectors().col(0).normalized();
  *normal = IRL::Normal(n[0], n[1], n[2]);
  normal->normalize();
  return true;
}

// Film thickness over the 3^3 stencil, in cell widths: the film's volume there
// over the area of the plane with normal n through its centroid, clipped to the
// stencil. Infinite without film, or if the plane misses the stencil.
inline double stencilThickness(const Phase& film, const int i, const int j, const int k, const IRL::Normal& n) {
  const Grid& grid = film.fields.grid;
  const double cell_volume = grid.h[0] * grid.h[1] * grid.h[2];
  double volume = 0.0, moment[3] = {0.0, 0.0, 0.0};
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) {
        const double f = film.fraction(ii, jj, kk);
        if (!(f > 0.0)) continue;
        const IRL::Pt& c = film.centroid(ii, jj, kk);
        volume += f * cell_volume;
        for (int d = 0; d < 3; ++d) moment[d] += f * cell_volume * c[d];
      }
  if (!(volume > 0.0)) return HUGE_VAL;
  const IRL::Pt centroid(moment[0] / volume, moment[1] / volume, moment[2] / volume);
  const IRL::RectangularCuboid stencil = IRL::RectangularCuboid::fromBoundingPts(
      IRL::Pt(grid.x(i - 1), grid.y(j - 1), grid.z(k - 1)), IRL::Pt(grid.x(i + 2), grid.y(j + 2), grid.z(k + 2)));
  const IRL::PlanarSeparator mid_plane = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, n * centroid));
  const double area = std::abs(
      IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(stencil, mid_plane, mid_plane[0]).calculateVolume());
  if (area <= 1.0e-12 * std::pow(cell_volume, 2.0 / 3.0)) return HUGE_VAL;
  return volume / area / std::cbrt(cell_volume);
}

// Very thin film: at most kPcaSlabThickness thick across the 3^3 stencil along
// its PCA normal. The stencil's largest cross-section is 3^2 sqrt(2) < 16 cell
// faces, so a film that thin holds less than 16 kPcaSlabThickness cell volumes
// there: a cheap test that skips the PCA for everything thicker.
inline bool isVeryThin(const Phase& film, const int i, const int j, const int k) {
  double volume = 0.0;
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) volume += film.fraction(ii, jj, kk);
  if (volume > 16.0 * kPcaSlabThickness) return false;
  IRL::Normal pca;
  return filmPcaNormal(film, i, j, k, &pca) && stencilThickness(film, i, j, k, pca) <= kPcaSlabThickness;
}

// ===========================================================================
// Topology of the other phase around a film (5^3 block)
// ===========================================================================

// Cells of the 5^3 block around (i,j,k), offsets -2..2: index and its inverse.
constexpr int kReach = 2, kBlockWidth = 2 * kReach + 1, kBlockCells = kBlockWidth * kBlockWidth * kBlockWidth;
inline int blockIndex(const int a, const int b, const int c) {
  return ((a + kReach) * kBlockWidth + (b + kReach)) * kBlockWidth + (c + kReach);
}
inline void blockOffsets(const int q, int* a, int* b, int* c) {
  *a = q / (kBlockWidth * kBlockWidth) - kReach;
  *b = (q / kBlockWidth) % kBlockWidth - kReach;
  *c = q % kBlockWidth - kReach;
}

// Floods the empty cells (no film) of the block connected through faces to
// `start`, marking them in `reached`; visit(q) is called for each.
template <class Visit>
inline void floodEmpty(const std::array<bool, kBlockCells>& empty, const int start,
                       std::array<bool, kBlockCells>* reached, Visit&& visit) {
  std::array<int, kBlockCells> stack;
  int top = 0;
  stack[top++] = start;
  (*reached)[start] = true;
  while (top > 0) {
    const int q = stack[--top];
    if (!visit(q)) return;
    int a, b, c;
    blockOffsets(q, &a, &b, &c);
    const int neighbours[6][3] = {{a - 1, b, c}, {a + 1, b, c}, {a, b - 1, c},
                                  {a, b + 1, c}, {a, b, c - 1}, {a, b, c + 1}};
    for (const auto& n : neighbours) {
      if (std::abs(n[0]) > kReach || std::abs(n[1]) > kReach || std::abs(n[2]) > kReach) continue;
      const int r = blockIndex(n[0], n[1], n[2]);
      if (!empty[r] || (*reached)[r]) continue;
      (*reached)[r] = true;
      stack[top++] = r;
    }
  }
}

// Thin-film guard: true where the other phase lies on both sides of the film,
// in separate regions, so the film continues through the cell and needs two
// planes. The block's empty cells form face-connected regions; it holds if at
// least two regions each come near the cell (within its 3^3 neighbourhood; for
// a thin film, within squared distance kGuardNearDistance2, since a thin film
// diagonal to the grid clips a third row of cells) and reach the block's edge
// (not an enclosed pocket). Drops, ligaments, resolved interfaces and film ends
// leave the other phase in one region.
inline bool filmContinues(const Phase& film, const int i, const int j, const int k) {
  std::array<bool, kBlockCells> empty, reached{};
  double volume_near = 0.0;
  for (int a = -kReach; a <= kReach; ++a)
    for (int b = -kReach; b <= kReach; ++b)
      for (int c = -kReach; c <= kReach; ++c) {
        const double f = film.fraction(i + a, j + b, k + c);
        empty[blockIndex(a, b, c)] = f <= VF_LOW;
        if (std::abs(a) <= 1 && std::abs(b) <= 1 && std::abs(c) <= 1) volume_near += f;
      }
  const int near_distance2 = volume_near <= kGuardThinVolume ? kGuardNearDistance2 : 3;
  int regions = 0;
  for (int start = 0; start < kBlockCells; ++start) {
    if (!empty[start] || reached[start]) continue;
    bool near = false, outer = false;
    floodEmpty(empty, start, &reached, [&](const int q) {
      int a, b, c;
      blockOffsets(q, &a, &b, &c);
      near = near || a * a + b * b + c * c <= near_distance2;
      outer = outer || std::max({std::abs(a), std::abs(b), std::abs(c)}) == kReach;
      return true;
    });
    if (near && outer && ++regions >= 2) return true;
  }
  return false;
}

// Mean normal of the two network faces (grid-index frame in, physical out);
// the longer one alone if either is shorter than 0.5.
inline IRL::Normal meanNormal(const IRL::Normal& n0, const IRL::Normal& n1, const Grid& grid) {
  const double a0 = n0.calculateMagnitude(), a1 = n1.calculateMagnitude();
  IRL::Normal m = (a0 >= 0.5 && a1 >= 0.5) ? n0 / a0 - n1 / a1 : (a0 >= a1 ? n0 : -n1);
  return toPhysical(m, grid);
}

// Topological edge sensor: does the other phase wrap around the film? Walking
// up to 2 cells from the cell along +-normal, the first empty cell on each side
// seeds a flood through empty cells: kEdge if the two seeds connect (the film
// ends within about 2 cells), else kContinuing. kThick if a side has no empty
// cell within 2 cells and the farthest cell walked there holds real film (a
// thin film cannot reach it; stray traces of film in the other phase do not
// count).
inline Edge edgeSensor(const Phase& film, const int i, const int j, const int k, const IRL::Normal& normal) {
  const Grid& grid = film.fields.grid;
  std::array<bool, kBlockCells> empty, reached{};
  std::array<double, kBlockCells> fraction;
  for (int a = -kReach; a <= kReach; ++a)
    for (int b = -kReach; b <= kReach; ++b)
      for (int c = -kReach; c <= kReach; ++c) {
        const int q = blockIndex(a, b, c);
        fraction[q] = film.fraction(i + a, j + b, k + c);
        empty[q] = fraction[q] <= VF_LOW;
      }
  double step_dir[3], longest = 0.0;   // the normal in cells, scaled to Chebyshev steps
  for (int d = 0; d < 3; ++d) {
    step_dir[d] = normal[d] / grid.h[d];
    longest = std::max(longest, std::abs(step_dir[d]));
  }
  if (!(longest > 0.0)) return kContinuing;
  int seed[2] = {-1, -1};
  bool thick = false;
  for (int side = 0; side < 2; ++side) {
    const double sign = side == 0 ? 1.0 : -1.0;
    double farthest_fraction = 0.0;
    for (int step = 1; step <= kReach && seed[side] < 0; ++step) {
      int c[3];
      for (int d = 0; d < 3; ++d) c[d] = static_cast<int>(std::lround(sign * step * step_dir[d] / longest));
      const int q = blockIndex(c[0], c[1], c[2]);
      if (empty[q]) seed[side] = q;
      else farthest_fraction = fraction[q];
    }
    if (seed[side] < 0 && farthest_fraction >= kThickFilmFraction) thick = true;
  }
  if (thick) return kThick;
  if (seed[0] < 0 || seed[1] < 0) return kContinuing;
  bool connected = false;
  floodEmpty(empty, seed[0], &reached, [&](const int q) { return !(connected = q == seed[1]); });
  return connected ? kEdge : kContinuing;
}

// ===========================================================================
// Network inputs
// ===========================================================================

// ML classifier class (1..6; 0: not classified) of the cell, the film phase
// counted as liquid: 5^3 volume fractions and fraction-weighted centroids
// relative to the cell centre, canonicalised by the classifier's own routine.
inline int classify(const Phase& film, const int i, const int j, const int k) {
  const Grid& grid = film.fields.grid;
  ml_classifier::Stencil stencil;
  const IRL::Pt centre = grid.cellCentre(i, j, k);
  for (int a = 0; a < 5; ++a)
    for (int b = 0; b < 5; ++b)
      for (int c = 0; c < 5; ++c) {
        const double f = film.fraction(i + a - 2, j + b - 2, k + c - 2);
        const IRL::Pt& centroid = film.centroid(i + a - 2, j + b - 2, k + c - 2);
        stencil.f(a, b, c) = f;
        for (int d = 0; d < 3; ++d) stencil.b(a, b, c, d) = (centroid[d] - centre[d]) / grid.h[d] * f;
      }
  if (stencil.f(ml_classifier::CID, ml_classifier::CID, ml_classifier::CID) < ml_classifier::EPSILON_CONNECT)
    return 0;
  double input[ml_classifier::NIN], logits[6];
  ml_classifier::detail::preprocess_and_flatten(stencil, input);
  classifierNetwork().evaluate(input, logits);
  return 1 + static_cast<int>(std::max_element(logits, logits + 6) - logits);
}

// The 189 PLICNet / R2P-Net moments: per 3^3 cell, the first phase's volume
// fraction and both phases' centroids (cell units, relative to their cell's
// centre). first_moment: the first phase's volume and first moments about the
// centre cell.
inline void stencilMoments(const Phase& first, const int i, const int j, const int k, double (&moments)[189],
                           double (&first_moment)[4]) {
  const Grid& grid = first.fields.grid;
  std::fill(first_moment, first_moment + 4, 0.0);
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) {
        double* m = moments + 7 * ((ii - i + 1) * 9 + (jj - j + 1) * 3 + (kk - k + 1));
        const IRL::Pt centre = grid.cellCentre(ii, jj, kk);
        const IRL::Pt& c0 = first.centroid(ii, jj, kk);
        const IRL::Pt& c1 = first.otherCentroid(ii, jj, kk);
        m[0] = first.fraction(ii, jj, kk);
        for (int d = 0; d < 3; ++d) {
          m[1 + d] = (c0[d] - centre[d]) / grid.h[d];
          m[4 + d] = (c1[d] - centre[d]) / grid.h[d];
        }
        first_moment[0] += m[0];
        first_moment[1] += (m[1] + (ii - i)) * m[0];
        first_moment[2] += (m[2] + (jj - j)) * m[0];
        first_moment[3] += (m[3] + (kk - k)) * m[0];
      }
}

// The canonicalisation applied to the moments (reflect_moments: a reflection
// `reflection`, then an axis permutation `permutation`), applied to a vector or
// undone on a network output.
inline void canonicalise(double (&v)[3], const int reflection, const int permutation) {
  static const bool flips[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                   {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
  for (int d = 0; d < 3; ++d)
    if (flips[reflection][d]) v[d] = -v[d];
  switch (permutation) {
    case 1: std::swap(v[0], v[1]); break;
    case 2: std::swap(v[1], v[2]); break;
    case 3: std::swap(v[0], v[2]); break;
    case 4: std::swap(v[0], v[1]); std::swap(v[1], v[2]); break;
    case 5: std::swap(v[0], v[1]); std::swap(v[0], v[2]); break;
  }
}
inline IRL::Normal uncanonicalise(const double* v, const int reflection, const int permutation) {
  IRL::Normal n(v[0], v[1], v[2]);
  switch (permutation) {
    case 1: std::swap(n[0], n[1]); break;
    case 2: std::swap(n[1], n[2]); break;
    case 3: std::swap(n[0], n[2]); break;
    case 4: std::swap(n[1], n[2]); std::swap(n[0], n[1]); break;
    case 5: std::swap(n[0], n[2]); std::swap(n[0], n[1]); break;
  }
  static const bool flips[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                   {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
  for (int d = 0; d < 3; ++d)
    if (flips[reflection][d]) n[d] = -n[d];
  return n;
}

// PLICNet's normal (physical, pointing out of the liquid). Its input puts the
// minority phase first, canonicalised about that phase's centre of mass.
inline IRL::Normal plicNetNormal(const Fields& fields, const int i, const int j, const int k) {
  const bool gas_first = fields.vf(i, j, k) >= 0.5;
  double moments[189], first_moment[4], output[3];
  stencilMoments(Phase{fields, gas_first}, i, j, k, moments, first_moment);
  const double centre_of_mass[3] = {first_moment[1] / first_moment[0], first_moment[2] / first_moment[0],
                                    first_moment[3] / first_moment[0]};
  int reflection = 0, permutation = 0;
  plicnet::reflect_moments(moments, centre_of_mass, &reflection, &permutation);
  plicNetwork().evaluate(moments, output);
  IRL::Normal n = uncanonicalise(output, reflection, permutation);
  if (!gas_first) n = -n;
  return toPhysical(n, fields.grid);
}

// R2P-Net's two face normals (grid-index frame, pointing into the film) and the
// PCA normal of the film's centroid cloud used to canonicalise its input.
struct NetworkFaces {
  IRL::Normal face[2];
  IRL::Normal pca;      // physical, oriented toward the film's centre of mass
  int pca_points = 0;   // cells in the centroid cloud
};
inline NetworkFaces r2pNetFaces(const Phase& film, const int i, const int j, const int k) {
  NetworkFaces result;
  double input[192], first_moment[4], output[6];
  double(&moments)[189] = *reinterpret_cast<double(*)[189]>(input);
  stencilMoments(film, i, j, k, moments, first_moment);
  Eigen::Vector3d points[27];
  result.pca_points = centroidCloud(film, i, j, k, points);
  const Eigen::Vector3d pca = cloudShape(points, result.pca_points).eigenvectors().col(0).normalized();
  result.pca = IRL::Normal(pca[0], pca[1], pca[2]);
  result.pca.normalize();
  if (IRL::dotProduct(result.pca, IRL::Pt(first_moment[1] / first_moment[0], first_moment[2] / first_moment[0],
                                          first_moment[3] / first_moment[0])) < 0)
    result.pca = -result.pca;
  double direction[3] = {result.pca[0], result.pca[1], result.pca[2]};
  int reflection = 0, permutation = 0;
  r2pnet::reflect_moments(moments, direction, &reflection, &permutation);
  canonicalise(direction, reflection, permutation);
  std::copy(direction, direction + 3, input + 189);
  r2pNetwork().evaluate(input, output);
  result.face[0] = uncanonicalise(output, reflection, permutation);
  result.face[1] = uncanonicalise(output + 3, reflection, permutation);
  return result;
}

// ===========================================================================
// Plane placement
// ===========================================================================

// Both planes shifted together until the cell holds the volume fraction vf.
// IRL's solver first; it can miss in nearly empty or full cells (vf ~ 1e-6),
// so a bisection on the shift follows if it does.
inline void matchVolume(const IRL::RectangularCuboid& cell, const double vf, const double tolerance,
                        IRL::PlanarSeparator* sep) {
  const double volume = cell.calculateVolume();
  auto error = [&](const IRL::PlanarSeparator& s) {
    return IRL::getVolumeMoments<IRL::VolumeMoments>(cell, s).volume() / volume - vf;
  };
  const IRL::PlanarSeparator start = *sep;
  IRL::setDistanceToMatchVolumeFraction(cell, vf, sep, tolerance);
  if (std::abs(error(*sep)) <= tolerance) return;
  auto shifted = [&](const double shift) {
    IRL::PlanarSeparator s = start;
    for (IRL::UnsignedIndex_t p = 0; p < s.getNumberOfPlanes(); ++p) s[p].distance() += shift;
    return s;
  };
  double lo = -std::cbrt(volume), hi = -lo;   // liquid volume grows with the shift
  while (error(shifted(lo)) > 0.0) lo *= 2.0;
  while (error(shifted(hi)) < 0.0) hi *= 2.0;
  for (int it = 0; it < 200 && hi - lo > DBL_EPSILON * (std::abs(lo) + std::abs(hi)); ++it) {
    const double mid = 0.5 * (lo + hi);
    *sep = shifted(mid);
    const double e = error(*sep);
    if (std::abs(e) <= tolerance) return;
    (e < 0.0 ? lo : hi) = mid;
  }
}

// Distances of two planes with fixed normals: Newton on (d0, d1) matching the
// volume fraction and the film's first moment across the film (along
// m = n0 - n1), with the film's own centroid as the target (the gas when the
// separator is flipped). Moving plane p sweeps its polygon, so dV/dd_p = A_p
// and dM/dd_p = A_p (m . centroid_p). Starts with both planes through the film
// centroid, keeps the best iterate, and finishes with a volume-matching shift.
inline void placeTwoPlanes(const IRL::RectangularCuboid& cell, const double vf, const IRL::Pt& liquid_centroid,
                           const IRL::Pt& gas_centroid, IRL::PlanarSeparator* sep) {
  constexpr double kTolerance = 1.0e-13;
  if (sep->getNumberOfPlanes() != 2) {
    IRL::setDistanceToMatchVolumeFraction(cell, vf, sep, kTolerance);
    return;
  }
  const double volume = cell.calculateVolume(), width = std::cbrt(volume);
  const IRL::Pt cell_centroid = cell.calculateCentroid();
  IRL::Normal n0 = (*sep)[0].normal(), n1 = (*sep)[1].normal();
  n0.normalize();
  n1.normalize();
  IRL::Normal m = n0 - n1;
  if (m.calculateMagnitude() < 1.0e-12) m = n0;
  m.normalize();

  const bool flipped = sep->isFlipped();
  const IRL::Pt& film_centroid = flipped ? gas_centroid : liquid_centroid;
  (*sep)[0] = IRL::Plane(n0, n0 * film_centroid);
  (*sep)[1] = IRL::Plane(n1, n1 * film_centroid);
  matchVolume(cell, vf, kTolerance, sep);

  // Residuals: volume error, and the liquid's first moment along m against its
  // target (built from the gas centroid when the film is gas).
  const double moment_target =
      flipped ? m * cell_centroid - (1.0 - vf) * (m * gas_centroid) : vf * (m * liquid_centroid);
  IRL::PlanarSeparator best = *sep;
  double best_merit = DBL_MAX;
  for (int it = 0; it < 20; ++it) {
    const auto moments = IRL::getVolumeMoments<IRL::VolumeMoments>(cell, *sep);
    const double r_volume = moments.volume() / volume - vf;
    const double r_moment = (m * moments.centroid() / volume - moment_target) / width;
    const double merit = r_volume * r_volume + r_moment * r_moment;
    if (merit < best_merit) {
      best_merit = merit;
      best = *sep;
    }
    if (std::abs(r_volume) < kTolerance && std::abs(r_moment) < kTolerance) break;
    double area[2], area_moment[2];
    for (int p = 0; p < 2; ++p) {
      const auto pm = IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, *sep, (*sep)[p]).calculateMoments();
      const double sign = pm.volume() < 0.0 ? -1.0 : 1.0;
      area[p] = sign * pm.volume() / volume;
      area_moment[p] = sign * (m * pm.centroid()) / (volume * width);
    }
    const double det = area[0] * area_moment[1] - area[1] * area_moment[0];
    if (std::abs(det) < 1.0e-14) break;   // a plane has left the cell
    const double step0 = -(area_moment[1] * r_volume - area[1] * r_moment) / det;
    const double step1 = -(area[0] * r_moment - area_moment[0] * r_volume) / det;
    // The residuals are only piecewise smooth: steps of at most one cell width.
    const double scale = std::max(1.0, std::max(std::abs(step0), std::abs(step1)) / width);
    (*sep)[0].distance() += step0 / scale;
    (*sep)[1].distance() += step1 / scale;
  }
  *sep = best;
  matchVolume(cell, vf, kTolerance, sep);
  IRL::cleanReconstruction(cell, vf, sep);
}

// ===========================================================================
// Pinch prevention
// ===========================================================================

// The film between two planes as half-spaces N.x <= D (N pointing out of the
// film), and its mean normal m.
struct FilmPlanes {
  IRL::Normal N[2], m;
  double D[2];
  explicit FilmPlanes(const IRL::PlanarSeparator& sep) {
    const double s = sep.isFlipped() ? -1.0 : 1.0;
    for (int p = 0; p < 2; ++p) {
      N[p] = s * sep[p].normal();
      D[p] = s * sep[p].distance();
    }
    m = N[0] - N[1];
    m.normalize();
  }
  // Film thickness along m through x (negative where the planes have crossed).
  double gapAt(const IRL::Pt& x) const {
    double top = std::numeric_limits<double>::infinity(), bottom = -top;
    for (int p = 0; p < 2; ++p) {
      const double nm = N[p] * m;
      if (std::abs(nm) < 1.0e-12) continue;
      const double s = (D[p] - N[p] * x) / nm;
      if (nm > 0.0) top = std::min(top, s);
      else bottom = std::max(bottom, s);
    }
    return top - bottom;
  }
  // Thinnest film inside the box [lo, hi]. The gap is linear, so its minimum is
  // at a vertex of box ∩ film: every intersection of three of the eight planes
  // (box faces and film planes) inside all eight. 0 where the planes meet.
  double minGap(const IRL::Pt& lo, const IRL::Pt& hi) const {
    IRL::Normal A[8];
    double B[8];
    for (int d = 0; d < 3; ++d) {
      A[2 * d] = IRL::Normal(d == 0, d == 1, d == 2);
      B[2 * d] = hi[d];
      A[2 * d + 1] = -A[2 * d];
      B[2 * d + 1] = -lo[d];
    }
    A[6] = N[0];
    B[6] = D[0];
    A[7] = N[1];
    B[7] = D[1];
    const double tolerance = 1.0e-10 * (hi[0] - lo[0]);
    double gap = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 8; ++a)
      for (int b = a + 1; b < 8; ++b)
        for (int c = b + 1; c < 8; ++c) {
          const IRL::Normal bc = IRL::crossProduct(A[b], A[c]);
          const double det = A[a] * bc;
          if (std::abs(det) < 1.0e-12) continue;
          const IRL::Normal x = IRL::Normal(B[a] * bc + B[b] * IRL::crossProduct(A[c], A[a]) +
                                            B[c] * IRL::crossProduct(A[a], A[b])) /
                                det;
          const IRL::Pt vertex(x[0], x[1], x[2]);
          bool inside = true;
          for (int q = 0; q < 8 && inside; ++q) inside = A[q] * vertex <= B[q] + tolerance;
          if (inside) gap = std::min(gap, gapAt(vertex));
        }
    return gap;
  }
};

// Two Newton-placed planes can meet inside the cell, giving the film zero
// thickness there: correct at a film's end, a pinch-off anywhere else. If the
// film is anywhere in the cell thinner than kPinchGapFraction times its mean
// thickness (film volume over the area of the plane through its centroid along
// m, clipped to the cell), both normals rotate toward m, each keeping its share
// of the opening, n(l) = normalize((n.m) m + (1 - l)(n - (n.m) m)), with the
// distances re-placed, by the smallest l in [0, 1] that removes the pinch
// (bisection; l = 1 is a parallel slab). Returns l.
inline double preventPinch(const IRL::RectangularCuboid& cell, const IRL::Pt& lo, const IRL::Pt& hi,
                           const double vf, const IRL::Pt& liquid_centroid, const IRL::Pt& gas_centroid,
                           IRL::PlanarSeparator* sep) {
  if (sep->getNumberOfPlanes() != 2) return 0.0;
  const bool flipped = sep->isFlipped();
  const IRL::Normal m = FilmPlanes(*sep).m, n0 = (*sep)[0].normal(), n1 = (*sep)[1].normal();
  const double width = hi[0] - lo[0], film_vf = flipped ? 1.0 - vf : vf, volume = cell.calculateVolume();
  double mean_thickness = film_vf * width;
  {
    const IRL::Pt& film_centroid = flipped ? gas_centroid : liquid_centroid;
    const IRL::PlanarSeparator mid_plane = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(m, m * film_centroid));
    const double area = std::abs(
        IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, mid_plane, mid_plane[0]).calculateVolume());
    if (area > 1.0e-12 * std::pow(volume, 2.0 / 3.0)) mean_thickness = film_vf * volume / area;
  }
  auto thick_enough = [&](const IRL::PlanarSeparator& s) {
    if (s.getNumberOfPlanes() != 2) return true;
    const double gap = FilmPlanes(s).minGap(lo, hi);
    return gap >= kPinchGapFraction * mean_thickness && gap > 1.0e-12 * width;
  };
  if (thick_enough(*sep)) return 0.0;

  auto rotated = [&](const IRL::Normal& n, const double l) {
    const IRL::Normal along = (n * m) * m;
    IRL::Normal r = along + (1.0 - l) * (n - along);
    r.normalize();
    return r;
  };
  auto placed = [&](const double l) {
    IRL::PlanarSeparator s = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(rotated(n0, l), 0.0), IRL::Plane(rotated(n1, l), 0.0), flipped ? -1.0 : 1.0);
    placeTwoPlanes(cell, vf, liquid_centroid, gas_centroid, &s);
    return s;
  };
  double l_lo = 0.0, l_hi = 1.0;
  for (int it = 0; it < 14; ++it) {
    const double mid = 0.5 * (l_lo + l_hi);
    (thick_enough(placed(mid)) ? l_hi : l_lo) = mid;
  }
  *sep = placed(l_hi);
  return l_hi;
}

// Pinch prevention is skipped at a film's end (the planes may meet there),
// unless the guard holds (it sees the film continue without relying on the
// network's normals, which the edge sensor's walk does), and for thick films
// (one cell's planes cannot pinch them off).
inline bool skipsPinchPrevention(const Edge edge, const bool guard) {
  return edge == kThick || (edge == kEdge && !guard);
}

// ===========================================================================
// One-plane choice
// ===========================================================================

// One plane with normal n at the cell's volume fraction, scored by how far its
// liquid and gas centroids land from the cell's; DBL_MAX if unusable.
inline double onePlaneScore(const Fields& fields, const int i, const int j, const int k,
                            const IRL::RectangularCuboid& cell, IRL::Normal n, IRL::PlanarSeparator* out) {
  if (n.calculateMagnitude() < 0.5) return DBL_MAX;
  n.normalize();
  const double vf = fields.vf(i, j, k);
  *out = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, IRL::findDistanceOnePlane(cell, vf, n)));
  const auto moments = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(cell, *out);
  if (std::abs(moments[0].volume() / cell.calculateVolume() - vf) > 1.0e-6) return DBL_MAX;
  double error = 0.0;
  if (vf > VF_LOW) error += IRL::magnitude(fields.liquid_centroid(i, j, k) - moments[0].centroid());
  if (vf < VF_HIGH) error += IRL::magnitude(fields.gas_centroid(i, j, k) - moments[1].centroid());
  return error;
}

// ===========================================================================
// Pass 2: joint fit of the two faces to the stencil's interface polygons
// ===========================================================================

struct InterfacePolygon {
  IRL::Pt centroid;
  IRL::Normal normal;
  double area = 0.0;
  int face = -1;   // 0 or 1; -1: neither
  bool in_centre_cell = false;
};

// Polygon of plane p of `sep` in the box [lo, hi], clipped to the part that
// bounds the phases (below the other planes; above them when flipped).
inline bool planePolygon(const IRL::Pt& lo, const IRL::Pt& hi, const IRL::PlanarSeparator& sep,
                         const IRL::UnsignedIndex_t p, InterfacePolygon* out) {
  const IRL::Normal& normal = sep[p].normal();
  const double distance = sep[p].distance();
  // Where the plane crosses the box's 12 edges.
  const double vx[2] = {lo[0], hi[0]}, vy[2] = {lo[1], hi[1]}, vz[2] = {lo[2], hi[2]};
  double px[8], py[8], pz[8], s[8];
  for (int v = 0; v < 8; ++v) {
    px[v] = vx[v >> 2];
    py[v] = vy[(v >> 1) & 1];
    pz[v] = vz[v & 1];
    s[v] = normal[0] * px[v] + normal[1] * py[v] + normal[2] * pz[v] - distance;
  }
  static const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                   {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  std::array<IRL::Pt, 12> crossings;
  int n_crossings = 0;
  for (const auto& e : edges) {
    const double s0 = s[e[0]], s1 = s[e[1]];
    if ((s0 <= 0.0 && s1 >= 0.0) || (s0 >= 0.0 && s1 <= 0.0)) {
      if (std::abs(s0 - s1) < 1.0e-14) continue;
      const double t = s0 / (s0 - s1);
      crossings[n_crossings++] = IRL::Pt(px[e[0]] + t * (px[e[1]] - px[e[0]]), py[e[0]] + t * (py[e[1]] - py[e[0]]),
                                         pz[e[0]] + t * (pz[e[1]] - pz[e[0]]));
    }
  }
  if (n_crossings < 3) return false;

  // Order the crossings around their mean.
  IRL::Normal t0 = IRL::crossProduct(normal, std::abs(normal[0]) < 0.9 ? IRL::Normal(1, 0, 0) : IRL::Normal(0, 1, 0));
  t0.normalize();
  IRL::Normal t1 = IRL::crossProduct(normal, t0);
  t1.normalize();
  double mean[3] = {0.0, 0.0, 0.0};
  for (int q = 0; q < n_crossings; ++q)
    for (int d = 0; d < 3; ++d) mean[d] += crossings[q][d];
  for (int d = 0; d < 3; ++d) mean[d] /= static_cast<double>(n_crossings);
  std::array<std::pair<double, int>, 12> order;
  for (int q = 0; q < n_crossings; ++q) {
    const IRL::Pt r(crossings[q][0] - mean[0], crossings[q][1] - mean[1], crossings[q][2] - mean[2]);
    order[q] = {std::atan2(t1 * r, t0 * r), q};
  }
  std::sort(order.begin(), order.begin() + n_crossings);
  double twice_area = 0.0;
  for (int q = 0; q < n_crossings; ++q) {
    const IRL::Pt& a = crossings[order[q].second];
    const IRL::Pt& b = crossings[order[(q + 1) % n_crossings].second];
    const double e1[3] = {a[0] - mean[0], a[1] - mean[1], a[2] - mean[2]};
    const double e2[3] = {b[0] - mean[0], b[1] - mean[1], b[2] - mean[2]};
    const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2],
                 cz = e1[0] * e2[1] - e1[1] * e2[0];
    twice_area += std::sqrt(cx * cx + cy * cy + cz * cz);
  }
  if (twice_area < 1.0e-30) return false;

  // Clip against the other planes (Sutherland-Hodgman).
  std::array<IRL::Pt, 16> vertices[2];
  int n_vertices = n_crossings, current = 0;
  for (int q = 0; q < n_crossings; ++q) vertices[0][q] = crossings[order[q].second];
  const double side = sep.isFlipped() ? -1.0 : 1.0;
  for (IRL::UnsignedIndex_t other = 0; other < sep.getNumberOfPlanes(); ++other) {
    if (other == p) continue;
    const IRL::Normal clip_normal = side * sep[other].normal();
    const double clip_distance = side * sep[other].distance();
    const std::array<IRL::Pt, 16>& in = vertices[current];
    std::array<IRL::Pt, 16>& kept = vertices[1 - current];
    int n_kept = 0;
    for (int q = 0; q < n_vertices; ++q) {
      const IRL::Pt& a = in[q];
      const IRL::Pt& b = in[(q + 1) % n_vertices];
      const double da = clip_normal * a - clip_distance, db = clip_normal * b - clip_distance;
      if (da <= 0.0 && n_kept < 16) kept[n_kept++] = a;
      if (((da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0)) && n_kept < 16) {
        const double t = da / (da - db);
        kept[n_kept++] = IRL::Pt(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]));
      }
    }
    n_vertices = n_kept;
    current = 1 - current;
    if (n_vertices < 3) return false;
  }

  // Area and centroid, as a fan of triangles from vertex 0.
  const std::array<IRL::Pt, 16>& v = vertices[current];
  double area = 0.0, centroid[3] = {0.0, 0.0, 0.0};
  for (int q = 1; q + 1 < n_vertices; ++q) {
    const double e1[3] = {v[q][0] - v[0][0], v[q][1] - v[0][1], v[q][2] - v[0][2]};
    const double e2[3] = {v[q + 1][0] - v[0][0], v[q + 1][1] - v[0][1], v[q + 1][2] - v[0][2]};
    const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2],
                 cz = e1[0] * e2[1] - e1[1] * e2[0];
    const double triangle = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    if (triangle <= 0.0) continue;
    area += triangle;
    for (int d = 0; d < 3; ++d) centroid[d] += triangle * (v[0][d] + v[q][d] + v[q + 1][d]) / 3.0;
  }
  if (area <= 0.0) return false;
  out->centroid = IRL::Pt(centroid[0] / area, centroid[1] / area, centroid[2] / area);
  out->normal = normal;
  out->normal.normalize();
  out->area = area;
  return true;
}

// `from` rotated toward `to` by at most max_angle (Rodrigues).
inline IRL::Normal limitedRotation(const IRL::Normal& from, const IRL::Normal& to, const double max_angle) {
  const double angle = std::acos(std::max(-1.0, std::min(1.0, from * to)));
  const double target = std::min(angle, max_angle);
  if (angle < 1.0e-12 || target < 1.0e-12) return from;
  IRL::Normal axis = IRL::crossProduct(from, to);
  if (axis.calculateMagnitude() < 1.0e-12) return from;
  axis.normalize();
  const double c = std::cos(target), s = std::sin(target);
  IRL::Normal out = from * c + IRL::crossProduct(axis, from) * s + axis * (axis * from) * (1.0 - c);
  out.normalize();
  return out;
}

// Joint least-squares fit of the two faces as height fields over one frame
// (the bisector of the centre cell's faces): face g is
//   h = a0_g + (a1 + d1_g) t + (a2 + d2_g) s + a3 t^2 + a4 t s + a5 s^2,
// a shared shape with separate offsets and ridged differences in slope.
// polygons[g][0] must be the centre cell's. False if the fit is rejected.
inline bool fitFaces(const InterfacePolygon* const polygons[2], const int n_polygons[2],
                     const IRL::Normal network[2], const double cell_size, IRL::Normal fitted[2]) {
  if (n_polygons[0] < kMinPolygonsPerFace || n_polygons[1] < kMinPolygonsPerFace) return false;
  IRL::Pt origin = polygons[0][0].centroid;
  IRL::Normal frame_normal = polygons[0][0].normal;
  {
    IRL::Normal n0 = polygons[0][0].normal, n1 = polygons[1][0].normal;
    n0.normalize();
    n1.normalize();
    const IRL::Normal bisector = 0.5 * n0 - 0.5 * n1;
    if (bisector.calculateMagnitude() >= kMinBisector) {
      frame_normal = bisector;
      origin = IRL::Pt(0.5 * polygons[0][0].centroid[0] + 0.5 * polygons[1][0].centroid[0],
                       0.5 * polygons[0][0].centroid[1] + 0.5 * polygons[1][0].centroid[1],
                       0.5 * polygons[0][0].centroid[2] + 0.5 * polygons[1][0].centroid[2]);
    }
  }
  IRL::Normal n_ref, t_ref, s_ref;
  plicparab::buildFrame(frame_normal, &n_ref, &t_ref, &s_ref);

  // Unknowns: [a0_0, a0_1, a1, a2, a3, a4, a5, d1_0, d2_0, d1_1, d2_1].
  constexpr int kUnknowns = 11, kMaxRows = 64;
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, kUnknowns, Eigen::ColMajor, kMaxRows, kUnknowns>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, kMaxRows, 1>;
  Matrix A = Matrix::Zero(kMaxRows, kUnknowns);
  Vector rhs = Vector::Zero(kMaxRows);
  int rows = 0, used[2] = {0, 0};
  double total_weight = 0.0;
  for (int g = 0; g < 2; ++g) {
    const IRL::Normal facing = g == 0 ? n_ref : -n_ref;
    for (int q = 0; q < n_polygons[g]; ++q) {
      const InterfacePolygon& polygon = polygons[g][q];
      IRL::Normal n = polygon.normal;
      n.normalize();
      const IRL::Pt r((polygon.centroid[0] - origin[0]) / cell_size, (polygon.centroid[1] - origin[1]) / cell_size,
                      (polygon.centroid[2] - origin[2]) / cell_size);
      const double t = t_ref * r, s = s_ref * r, h = n_ref * r;
      const double kernel = plicparab::wgauss(std::sqrt(t * t + s * s + h * h), kFitRadius);
      if (kernel <= 0.0) continue;
      const double weight = polygon.area / (cell_size * cell_size) * std::max(n * facing, 0.0) * kernel;
      if (weight <= 0.0 || rows >= kMaxRows - 7) continue;
      const double sw = std::sqrt(weight);
      A(rows, g) = sw;
      A(rows, 2) = sw * t;
      A(rows, 3) = sw * s;
      A(rows, 4) = sw * t * t;
      A(rows, 5) = sw * t * s;
      A(rows, 6) = sw * s * s;
      A(rows, 7 + 2 * g) = sw * t;
      A(rows, 8 + 2 * g) = sw * s;
      rhs(rows) = sw * h;
      total_weight += weight;
      ++used[g];
      ++rows;
    }
  }
  if (used[0] < kMinPolygonsPerFace || used[1] < kMinPolygonsPerFace) return false;
  const int data_rows = rows;
  const double splay_weight = std::sqrt(kSplayPenalty * total_weight);
  const double curvature_weight = std::sqrt(kCurvaturePenalty * total_weight);
  for (int c = 7; c < kUnknowns; ++c) A(rows++, c) = splay_weight;
  for (int c = 4; c < 7; ++c) A(rows++, c) = curvature_weight;
  const Matrix system = A.topRows(rows);
  const Eigen::Matrix<double, kUnknowns, 1> solution = system.colPivHouseholderQr().solve(rhs.head(rows));
  if (!solution.allFinite()) return false;

  double squared_residual = 0.0;
  for (int r = 0; r < data_rows; ++r) {
    const double residual = A.row(r).dot(solution.transpose()) - rhs(r);
    squared_residual += residual * residual;
  }
  if (std::sqrt(squared_residual / static_cast<double>(data_rows)) > kMaxFitResidual) return false;

  for (int g = 0; g < 2; ++g) {
    const double slope_t = solution(2) + solution(7 + 2 * g), slope_s = solution(3) + solution(8 + 2 * g);
    IRL::Normal f = n_ref - slope_t * t_ref - slope_s * s_ref;
    f.normalize();
    if (f * network[g] < 0.0) f = -f;
    fitted[g] = f;
  }
  return true;
}

// Refined normals for the two-plane cell (i,j,k), or false to keep pass 1's.
inline bool refineNormals(const Fields& fields, const Field<IRL::PlanarSeparator>& interface, const int i,
                          const int j, const int k, IRL::Normal normal[2]) {
  const Grid& grid = fields.grid;
  const double cell_size = (grid.h[0] + grid.h[1] + grid.h[2]) / 3.0;

  // Interface polygons of the 3^3 stencil, the centre cell's first; per cell.
  std::array<InterfacePolygon, 54> polygons;
  std::array<int, 28> cell_begin;
  int n_polygons = 0, n_cells = 0;
  cell_begin[0] = 0;
  for (int centre_pass = 1; centre_pass >= 0; --centre_pass)
    for (int ii = i - 1; ii <= i + 1; ++ii)
      for (int jj = j - 1; jj <= j + 1; ++jj)
        for (int kk = k - 1; kk <= k + 1; ++kk) {
          const bool is_centre = ii == i && jj == j && kk == k;
          if (is_centre != (centre_pass == 1)) continue;
          const IRL::PlanarSeparator& sep = interface(ii, jj, kk);
          const double f = fields.vf(ii, jj, kk);
          if (f > VF_LOW && f < VF_HIGH) {
            const IRL::Pt lo(grid.x(ii), grid.y(jj), grid.z(kk)), hi(grid.x(ii + 1), grid.y(jj + 1), grid.z(kk + 1));
            for (IRL::UnsignedIndex_t p = 0; p < sep.getNumberOfPlanes() && n_polygons < 54; ++p) {
              if (!planePolygon(lo, hi, sep, p, &polygons[n_polygons])) continue;
              polygons[n_polygons].face = -1;
              polygons[n_polygons].in_centre_cell = is_centre;
              ++n_polygons;
            }
          }
          cell_begin[++n_cells] = n_polygons;
        }
  if (n_polygons < 2 * kMinPolygonsPerFace) return false;

  // Sort the polygons into the two faces by orientation, a cell's two planes jointly.
  const IRL::Normal net0 = normal[0], net1 = normal[1];
  auto assign_single = [&](InterfacePolygon& polygon) {
    const double d0 = polygon.normal * net0, d1 = polygon.normal * net1;
    polygon.face = std::max(d0, d1) > 0.0 ? (d0 >= d1 ? 0 : 1) : -1;
  };
  for (int c = 0; c < n_cells; ++c) {
    const int begin = cell_begin[c], end = cell_begin[c + 1];
    if (end - begin == 1) {
      assign_single(polygons[begin]);
    } else if (end - begin >= 2) {
      InterfacePolygon& a = polygons[begin];
      InterfacePolygon& b = polygons[begin + 1];
      const double a0 = a.normal * net0, a1 = a.normal * net1, b0 = b.normal * net0, b1 = b.normal * net1;
      if (a0 + b1 >= a1 + b0) {
        a.face = a0 > 0.0 ? 0 : -1;
        b.face = b1 > 0.0 ? 1 : -1;
      } else {
        a.face = a1 > 0.0 ? 1 : -1;
        b.face = b0 > 0.0 ? 0 : -1;
      }
      for (int extra = begin + 2; extra < end; ++extra) assign_single(polygons[extra]);
    }
  }
  double face_area[2] = {0.0, 0.0}, total_area = 0.0;
  for (int q = 0; q < n_polygons; ++q) {
    if (polygons[q].face >= 0) face_area[polygons[q].face] += polygons[q].area;
    total_area += polygons[q].area;
  }
  if (total_area <= 0.0) return false;
  if (std::min(face_area[0], face_area[1]) / total_area < kMinFaceAreaFraction) return false;

  // Each face's polygons, the centre cell's first.
  std::array<InterfacePolygon, 54> by_face[2];
  int n_by_face[2] = {0, 0};
  for (int g = 0; g < 2; ++g) {
    for (int centre_first = 1; centre_first >= 0; --centre_first)
      for (int q = 0; q < n_polygons; ++q)
        if (polygons[q].face == g && polygons[q].in_centre_cell == (centre_first == 1))
          by_face[g][n_by_face[g]++] = polygons[q];
    if (n_by_face[g] == 0 || !by_face[g][0].in_centre_cell) return false;
  }
  const InterfacePolygon* const face_polygons[2] = {by_face[0].data(), by_face[1].data()};
  const IRL::Normal network[2] = {net0, net1};
  IRL::Normal fitted[2];
  if (!fitFaces(face_polygons, n_by_face, network, cell_size, fitted)) return false;

  const IRL::Normal limited0 = limitedRotation(net0, fitted[0], kMaxRotation);
  const IRL::Normal limited1 = limitedRotation(net1, fitted[1], kMaxRotation);
  const double splay_before = std::acos(std::max(-1.0, std::min(1.0, -(net0 * net1))));
  const double splay_after = std::acos(std::max(-1.0, std::min(1.0, -(limited0 * limited1))));
  if (std::abs(splay_after - splay_before) > kMaxSplayChange) return false;
  normal[0] = limited0;
  normal[1] = limited1;
  return true;
}

inline double centroidError(const IRL::RectangularCuboid& cell, const IRL::PlanarSeparator& sep, const int phase,
                            const IRL::Pt& target) {
  const auto moments = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>,
                                                       IRL::ReconstructionDefaultCuttingMethod>(cell, sep);
  return IRL::magnitude(moments[phase].centroid() - target);
}

// Two exactly opposite normals: a slab (the distance solve changes distances only).
inline bool isSlab(const IRL::PlanarSeparator& sep) {
  if (sep.getNumberOfPlanes() != 2) return false;
  const IRL::Normal &a = sep[0].normal(), &b = sep[1].normal();
  return a[0] == -b[0] && a[1] == -b[1] && a[2] == -b[2];
}

// ===========================================================================
// Ghost cells
// ===========================================================================

// Periodic copies of the interior planes, shifted by one domain length.
inline void fillGhostPlanes(Field<IRL::PlanarSeparator>* interface) {
  interface->fillGhosts();
  const Grid& grid = interface->grid();
  FOR_ALL(grid, i, j, k) {
    if (!grid.isGhost(i, j, k)) continue;
    const int index[3] = {i, j, k};
    for (auto& plane : (*interface)(i, j, k))
      for (int d = 0; d < 3; ++d) {
        if (index[d] < 0) plane.distance() -= plane.normal()[d] * grid.length(d);
        if (index[d] >= grid.n[d]) plane.distance() += plane.normal()[d] * grid.length(d);
      }
  }
}

// ===========================================================================
// The reconstruction
// ===========================================================================

// Per two-plane cell, what pass 2 needs from pass 1.
struct FilmCell {
  int i, j, k;
  Edge edge;
  bool guard;
  Slab slab;
};

// Reconstructs every interior cell's planes from the volume fraction and the
// phase centroids (ghost cells filled), then fills the ghost planes.
// `diagnostics` may be null.
inline void reconstruct(const Field<double>& vf, const Field<IRL::Pt>& liquid_centroid,
                        const Field<IRL::Pt>& gas_centroid, Field<IRL::PlanarSeparator>* interface,
                        Diagnostics* diagnostics = nullptr) {
  const Grid& grid = vf.grid();
  const Fields fields{grid, vf, liquid_centroid, gas_centroid};
  auto record = [&](Field<int> Diagnostics::*field, const int i, const int j, const int k, const int value) {
    if (diagnostics != nullptr) (diagnostics->*field)(i, j, k) = value;
  };
  auto record_unpinch = [&](const int i, const int j, const int k, const double l) {
    if (diagnostics != nullptr) diagnostics->unpinch(i, j, k) = std::max(diagnostics->unpinch(i, j, k), l);
  };

  // ---- Pass 1, cell by cell -------------------------------------------------
  std::vector<FilmCell> two_plane_cells;
  FOR_INTERIOR(grid, i, j, k) {
    IRL::PlanarSeparator& sep = (*interface)(i, j, k);
    const double f = vf(i, j, k);
    record(&Diagnostics::classifier, i, j, k, 0);
    record(&Diagnostics::guard, i, j, k, 0);
    record(&Diagnostics::slab, i, j, k, kNoSlab);
    record(&Diagnostics::edge, i, j, k, kNotComputed);
    if (diagnostics != nullptr) diagnostics->unpinch(i, j, k) = 0.0;
    if (f < VF_LOW || f > VF_HIGH) {   // pure cell: no plane
      sep = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), std::copysign(1.0, f - 0.5)));
      record(&Diagnostics::route, i, j, k, kPure);
      continue;
    }
    const IRL::RectangularCuboid cell = grid.cell(i, j, k);
    const IRL::Pt &liquid = liquid_centroid(i, j, k), &gas = gas_centroid(i, j, k);

    // Film phase and routing.
    const Phase film{fields, filmIsGas(fields, i, j, k)};
    const int cls = classify(film, i, j, k);
    const bool guard = filmContinues(film, i, j, k);
    const bool use_r2p = cls == kSheetClass || cls == kSheetEndClass || guard || isVeryThin(film, i, j, k);
    record(&Diagnostics::classifier, i, j, k, cls);
    if (guard) record(&Diagnostics::guard, i, j, k, cls == kSheetClass || cls == kSheetEndClass ? 2 : 1);
    record(&Diagnostics::route, i, j, k, use_r2p ? kR2PNet : kPlicNet);

    if (!use_r2p) {   // PLICNet
      const IRL::Normal n = plicNetNormal(fields, i, j, k);
      sep = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, IRL::findDistanceOnePlane(cell, f, n)));
      continue;
    }

    // R2P-Net's faces, or a slab where they cannot be trusted.
    NetworkFaces net = r2pNetFaces(film, i, j, k);
    IRL::Normal &n0 = net.face[0], &n1 = net.face[1];
    Slab slab = kNoSlab;
    if (net.pca_points >= 6 && stencilThickness(film, i, j, k, net.pca) <= kPcaSlabThickness) {
      n0 = IRL::Normal(net.pca[0] / grid.h[0], net.pca[1] / grid.h[1], net.pca[2] / grid.h[2]);
      n0.normalize();
      n1 = -n0;
      slab = kPcaSlab;
    }
    const Edge edge = edgeSensor(film, i, j, k, meanNormal(n0, n1, grid));
    bool one_plane = n1.calculateMagnitude() < kTwoPlaneMagnitude || n0.calculateMagnitude() < kTwoPlaneMagnitude;
    if (one_plane && guard && std::max(n0.calculateMagnitude(), n1.calculateMagnitude()) > 0.0) {
      // The film continues: a slab around the longer normal.
      if (n0.calculateMagnitude() >= n1.calculateMagnitude()) n1 = -n0;
      else n0 = -n1;
      one_plane = false;
      slab = kGuardSlab;
    }
    record(&Diagnostics::slab, i, j, k, slab);
    record(&Diagnostics::edge, i, j, k, edge);

    if (!one_plane) {
      // Two planes around the film. The network's normals point into the film;
      // the separator's point out of the liquid, which is the film unless the
      // film is gas (then the separator is flipped).
      IRL::Normal p0 = toPhysical(n0, grid), p1 = toPhysical(n1, grid);
      if (!film.is_gas) {
        p0 = -p0;
        p1 = -p1;
      }
      sep = IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(p0, 0), IRL::Plane(p1, 0), film.is_gas ? -1 : 1);
      placeTwoPlanes(cell, f, liquid, gas, &sep);
      if (!skipsPinchPrevention(edge, guard)) {
        const IRL::Pt lo(grid.x(i), grid.y(j), grid.z(k)), hi(grid.x(i + 1), grid.y(j + 1), grid.z(k + 1));
        record_unpinch(i, j, k, preventPinch(cell, lo, hi, f, liquid, gas, &sep));
      }
      if (sep.getNumberOfPlanes() == 2) two_plane_cells.push_back({i, j, k, edge, guard, slab});
      continue;
    }

    // One plane: the network's surviving normal or PLICNet's, whichever puts
    // the phase centroids closer to the cell's.
    IRL::Normal n = n1.calculateMagnitude() < n0.calculateMagnitude() ? n0 : n1;
    for (int d = 0; d < 3; ++d) n[d] *= grid.h[d];
    if (n.calculateMagnitude() > 0.0) n.normalize();
    if (!film.is_gas) n = -n;
    if (IRL::dotProduct(n, liquid - cell.calculateCentroid()) > 0) n = -n;
    IRL::PlanarSeparator from_network, from_plicnet;
    const double error_network = onePlaneScore(fields, i, j, k, cell, n, &from_network);
    const double error_plicnet = onePlaneScore(fields, i, j, k, cell, plicNetNormal(fields, i, j, k), &from_plicnet);
    sep = error_plicnet < error_network ? from_plicnet : from_network;
  }
  fillGhostPlanes(interface);

  // ---- Pass 2: refine the two-plane cells against their neighbours ---------
  // Every cell reads the pass-1 planes; the results are written afterwards.
  std::vector<std::pair<const FilmCell*, IRL::PlanarSeparator>> refined;
  refined.reserve(two_plane_cells.size());
  for (const FilmCell& c : two_plane_cells) {
    const IRL::PlanarSeparator& pass1 = (*interface)(c.i, c.j, c.k);
    IRL::Normal normal[2] = {pass1[0].normal(), pass1[1].normal()};
    normal[0].normalize();
    normal[1].normalize();
    if (!refineNormals(fields, *interface, c.i, c.j, c.k, normal)) continue;
    if (isSlab(pass1)) {   // a slab may rotate but stays parallel
      IRL::Normal mean = normal[0] - normal[1];
      mean.normalize();
      normal[0] = mean;
      normal[1] = -mean;
    }
    const double f = vf(c.i, c.j, c.k);
    const IRL::Pt &liquid = liquid_centroid(c.i, c.j, c.k), &gas = gas_centroid(c.i, c.j, c.k);
    const IRL::RectangularCuboid cell = grid.cell(c.i, c.j, c.k);
    const double flip = pass1.isNotFlipped() ? 1.0 : -1.0;
    IRL::PlanarSeparator sep =
        IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(normal[0], 0.0), IRL::Plane(normal[1], 0.0), flip);
    placeTwoPlanes(cell, f, liquid, gas, &sep);
    if (!skipsPinchPrevention(c.edge, c.guard)) {
      const IRL::Pt lo(grid.x(c.i), grid.y(c.j), grid.z(c.k)), hi(grid.x(c.i + 1), grid.y(c.j + 1), grid.z(c.k + 1));
      record_unpinch(c.i, c.j, c.k, preventPinch(cell, lo, hi, f, liquid, gas, &sep));
    }
    // One plane instead, if it matches the film centroid about as well; never
    // in a film the guard holds for or a very thin film's slab.
    if (sep.getNumberOfPlanes() == 2 && !c.guard && c.slab != kPcaSlab) {
      const int film_phase = flip < 0.0 ? 1 : 0;
      const IRL::Pt& target = film_phase == 1 ? gas : liquid;
      const double error_two = centroidError(cell, sep, film_phase, target);
      double error_one = -1.0;
      IRL::PlanarSeparator best_one;
      for (int g = 0; g < 2; ++g) {
        IRL::PlanarSeparator candidate = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal[g], 0.0));
        IRL::setDistanceToMatchVolumeFraction(cell, f, &candidate);
        const double e = centroidError(cell, candidate, film_phase, target);
        if (error_one < 0.0 || e < error_one) {
          error_one = e;
          best_one = candidate;
        }
      }
      if (error_one >= 0.0 && error_one <= kPlaneDropBias * error_two) sep = best_one;
    }
    refined.emplace_back(&c, sep);
  }
  for (const auto& r : refined) (*interface)(r.first->i, r.first->j, r.first->k) = r.second;
  fillGhostPlanes(interface);
}

}  // namespace r2p

#endif  // EXAMPLES_R2P_ADVECTOR_R2P_NET_H_
