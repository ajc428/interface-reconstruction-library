// Topological film-edge sensor: does the other phase reach around the film?
//
// Above and below a film, the other phase forms two regions that meet only
// where the film ends. Every cell a film passes through holds some film, and
// the cells cut by a surface separate the cells on its two sides when those
// are joined through faces only. So a continuing film -- however thin, tilted
// or curved -- keeps the two regions apart, and so does a thick rim.
//
// The sensor seeds one empty cell on each side of the film, walking from the
// cell along +-m (R2P-Net's mean normal) up to kReach cells, then floods the
// empty cells of the (2 kReach + 1)^3 block around the cell through faces from
// one seed. The cell is an edge if the flood reaches the other seed: the film
// ends within about kReach cells. Empty: film-phase VF <= VF_LOW.
//
// Alternative to r2p_edge_sensor.h (presence-based, left as it was); which one
// gates pinch prevention: useTopology(). The Fortran port is
// r2p_edge_topology in r2p_net_tools.f90.
//
// The same idea guards the R2P routing (filmSeparates): where the other phase
// lies on both sides of the film, apart, the film continues through the cell
// and needs two planes, whatever the classifier or detector says.

#ifndef EXAMPLES_NEW_ADVECTOR_R2P_EDGE_TOPOLOGY_H_
#define EXAMPLES_NEW_ADVECTOR_R2P_EDGE_TOPOLOGY_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "irl/geometry/general/normal.h"
#include "irl/parameters/constants.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"

namespace r2pedgetopo {

constexpr int kReach = 2;                // block half-width and seed search distance, cells
constexpr int kWidth = 2 * kReach + 1;
constexpr int kCells = kWidth * kWidth * kWidth;
constexpr double kThickFilm = 0.01;      // film VF in the farthest walked cell of a blocked side, for "thick"

// Pinch prevention skips the cells this sensor calls edges, unless
// R2P_EDGE_SENSOR=presence selects r2p_edge_sensor.h instead.
inline bool useTopology() {
  static const bool v = [] {
    const char* s = std::getenv("R2P_EDGE_SENSOR");
    return !(s != nullptr && std::strcmp(s, "presence") == 0);
  }();
  return v;
}

// 1 if the other phase on the two sides of the film around (i,j,k) connects
// within the block (an edge), 0 if not, 2 if a side has no empty cell within
// kReach cells and the farthest cell walked there holds film VF >= kThickFilm
// (a film thicker than that, e.g. a thick rounded rim: no thin film to keep
// from pinching off), -1 where the block reaches past the ghost cells. A thin
// film cannot reach that far cell, so a side blocked only by stray traces of
// film in the other phase gives 0: the film is thin. normal: the film's mean
// normal (unit, physical).
inline int gasWraps(const Data<double>& vf, const bool film_is_gas, const int i, const int j, const int k,
                    const IRL::Normal& normal) {
  const BasicMesh& mesh = vf.getMesh();
  if (i - kReach < mesh.imino() || i + kReach > mesh.imaxo() || j - kReach < mesh.jmino() ||
      j + kReach > mesh.jmaxo() || k - kReach < mesh.kmino() || k + kReach > mesh.kmaxo())
    return -1;
  auto index = [](const int a, const int b, const int c) {
    return ((a + kReach) * kWidth + (b + kReach)) * kWidth + (c + kReach);
  };
  std::array<bool, kCells> empty;
  std::array<double, kCells> film;
  for (int a = -kReach; a <= kReach; ++a)
    for (int b = -kReach; b <= kReach; ++b)
      for (int c = -kReach; c <= kReach; ++c) {
        const double f = film_is_gas ? 1.0 - vf(i + a, j + b, k + c) : vf(i + a, j + b, k + c);
        film[index(a, b, c)] = f;
        empty[index(a, b, c)] = f <= IRL::global_constants::VF_LOW;
      }

  // Seeds: the first empty cell walking from the cell along +normal and along
  // -normal, in Chebyshev steps of one cell.
  const double h[3] = {mesh.dx(), mesh.dy(), mesh.dz()};
  double d[3], dmax = 0.0;
  for (int a = 0; a < 3; ++a) {
    d[a] = normal[a] / h[a];
    dmax = std::max(dmax, std::abs(d[a]));
  }
  if (!(dmax > 0.0)) return 0;
  int seed[2] = {-1, -1};
  bool thick = false;
  for (int side = 0; side < 2; ++side) {
    const double sign = side == 0 ? 1.0 : -1.0;
    double far = 0.0;   // film in the farthest cell walked on this side
    for (int step = 1; step <= kReach && seed[side] < 0; ++step) {
      int c[3];
      for (int a = 0; a < 3; ++a) c[a] = static_cast<int>(std::lround(sign * step * d[a] / dmax));
      if (empty[index(c[0], c[1], c[2])]) seed[side] = index(c[0], c[1], c[2]);
      else far = film[index(c[0], c[1], c[2])];
    }
    if (seed[side] < 0 && far >= kThickFilm) thick = true;
  }
  if (thick) return 2;
  if (seed[0] < 0 || seed[1] < 0) return 0;

  // Flood the empty cells through faces from the first seed.
  std::array<bool, kCells> reached{};
  std::array<int, kCells> stack;
  int top = 0;
  stack[top++] = seed[0];
  reached[seed[0]] = true;
  while (top > 0) {
    const int q = stack[--top];
    if (q == seed[1]) return 1;
    const int a = q / (kWidth * kWidth) - kReach, b = (q / kWidth) % kWidth - kReach, c = q % kWidth - kReach;
    const int nb[6][3] = {{a - 1, b, c}, {a + 1, b, c}, {a, b - 1, c}, {a, b + 1, c}, {a, b, c - 1}, {a, b, c + 1}};
    for (const auto& n : nb) {
      if (std::abs(n[0]) > kReach || std::abs(n[1]) > kReach || std::abs(n[2]) > kReach) continue;
      const int r = index(n[0], n[1], n[2]);
      if (!empty[r] || reached[r]) continue;
      reached[r] = true;
      stack[top++] = r;
    }
  }
  return 0;
}

// Thin-film guard for the R2P routing: true where the other phase lies on both
// sides of the film around (i,j,k), in separate regions. The empty cells of
// the (2 kReach + 1)^3 block (film-phase VF <= VF_LOW) are split into regions
// joined through faces; the film is thin and continues here if at least two
// regions each reach both the cell's 3^3 neighbourhood (the other phase within
// about a cell of it on that side) and the edge of the block (not an enclosed
// pocket). A droplet, a ligament or a film's end leaves the other phase in one
// region around it; a resolved interface has it on one side only. Films up to
// about a cell thick qualify (a diagonal one up to ~0.9 cells). Where the film
// in the 3^3 neighbourhood is thin (volume <= kGuardThinVolume), "near" reaches
// to squared distance kGuardNearDist2: a thin film diagonal to the grid clips
// the corners of a third row of cells, which can fill the 3^3 neighbourhood on
// one side and leave the other phase there only at the diagonal (0,1,2)
// neighbour. Fortran: r2p_film_guard in r2p_net_tools.f90.
constexpr double kGuardThinVolume = 1.0;   // film volume in the 3^3 neighbourhood, cells
constexpr int kGuardNearDist2 = 5;         // squared distance, cells

inline bool filmSeparates(const Data<double>& vf, const bool film_is_gas, const int i, const int j, const int k) {
  const BasicMesh& mesh = vf.getMesh();
  if (i - kReach < mesh.imino() || i + kReach > mesh.imaxo() || j - kReach < mesh.jmino() ||
      j + kReach > mesh.jmaxo() || k - kReach < mesh.kmino() || k + kReach > mesh.kmaxo())
    return false;
  auto index = [](const int a, const int b, const int c) {
    return ((a + kReach) * kWidth + (b + kReach)) * kWidth + (c + kReach);
  };
  std::array<bool, kCells> empty, done{};
  double near_volume = 0.0;
  for (int a = -kReach; a <= kReach; ++a)
    for (int b = -kReach; b <= kReach; ++b)
      for (int c = -kReach; c <= kReach; ++c) {
        const double f = film_is_gas ? 1.0 - vf(i + a, j + b, k + c) : vf(i + a, j + b, k + c);
        empty[index(a, b, c)] = f <= IRL::global_constants::VF_LOW;
        if (std::abs(a) <= 1 && std::abs(b) <= 1 && std::abs(c) <= 1) near_volume += f;
      }
  const int near_dist2 = near_volume <= kGuardThinVolume ? kGuardNearDist2 : 3;   // 3: exactly the 3^3 neighbourhood
  std::array<int, kCells> stack;
  int regions = 0;
  for (int start = 0; start < kCells; ++start) {
    if (!empty[start] || done[start]) continue;
    // Flood one region through faces; does it come near the cell and reach the block's edge?
    bool near = false, outer = false;
    int top = 0;
    stack[top++] = start;
    done[start] = true;
    while (top > 0) {
      const int q = stack[--top];
      const int a = q / (kWidth * kWidth) - kReach, b = (q / kWidth) % kWidth - kReach, c = q % kWidth - kReach;
      const int reach = std::max({std::abs(a), std::abs(b), std::abs(c)});
      near = near || a * a + b * b + c * c <= near_dist2;
      outer = outer || reach == kReach;
      const int nb[6][3] = {{a - 1, b, c}, {a + 1, b, c}, {a, b - 1, c}, {a, b + 1, c}, {a, b, c - 1}, {a, b, c + 1}};
      for (const auto& n : nb) {
        if (std::abs(n[0]) > kReach || std::abs(n[1]) > kReach || std::abs(n[2]) > kReach) continue;
        const int r = index(n[0], n[1], n[2]);
        if (!empty[r] || done[r]) continue;
        done[r] = true;
        stack[top++] = r;
      }
    }
    if (near && outer && ++regions >= 2) return true;
  }
  return false;
}

}  // namespace r2pedgetopo

#endif  // EXAMPLES_NEW_ADVECTOR_R2P_EDGE_TOPOLOGY_H_
