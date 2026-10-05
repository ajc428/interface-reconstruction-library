// Film-edge sensor: C++ version of NGA2's detect_edge_regions.
//
// Probes 16 directions, 22.5 deg apart, in the film's tangent plane. In each,
// the neighbour best aligned with it is taken on two rings -- the 26 cells
// around (distance 1) and the shell of the 5^3 block (distance 2) -- and the
// direction counts as empty if little film is near that neighbour:
//   presence(c) = sum over the 3^3 cells around c of min(f_n / f_centre, 1)
// (f = film-phase volume fraction, cells with film only), empty when
// presence <= 3 on the inner ring or <= 0.25 on the outer one. Returns the
// number of distinct empty probe cells; NGA2 calls >= 2 an edge.
//
// Changes from the Fortran: the film phase (not the liquid) is measured; the
// tangent plane is R2P-Net's mean normal (n0 - n1) rather than one plane of
// the previous reconstruction, and its basis never degenerates; each probe
// cell is counted once per ring, however many directions select it.

#ifndef EXAMPLES_NEW_ADVECTOR_R2P_EDGE_SENSOR_H_
#define EXAMPLES_NEW_ADVECTOR_R2P_EDGE_SENSOR_H_

#include <algorithm>
#include <array>
#include <cmath>

#include "irl/geometry/general/normal.h"
#include "irl/parameters/constants.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"

namespace r2pedge {

constexpr int kDirections = 16;
constexpr double kInnerEmpty = 3.0;    // presence at or below: empty (distance-1 ring)
constexpr double kOuterEmpty = 0.25;   // presence at or below: empty (distance-2 shell)
constexpr double kEdgeMinCount = 2.0;  // this many empty probe cells or more: an edge

// The film's mean normal from R2P-Net's two face normals (grid frame, before
// mesh scaling), or the surviving one when the network predicts one plane.
inline IRL::Normal meanNormal(IRL::Normal n0, IRL::Normal n1, const BasicMesh& mesh) {
  const double a0 = n0.calculateMagnitude(), a1 = n1.calculateMagnitude();
  IRL::Normal m;
  if (a0 >= 0.5 && a1 >= 0.5) m = n0 / a0 - n1 / a1;
  else m = a0 >= a1 ? n0 : -n1;
  m[0] *= mesh.dx();
  m[1] *= mesh.dy();
  m[2] *= mesh.dz();
  m.normalize();
  return m;
}

// Number of distinct empty probe cells around (i,j,k); -1 where the probes'
// 3^3 sums would reach past the ghost cells.
inline int edgeCount(const Data<double>& vf, const bool film_is_gas, const int i, const int j,
                     const int k, const IRL::Normal& normal) {
  const BasicMesh& mesh = vf.getMesh();
  if (i - 3 < mesh.imino() || i + 3 > mesh.imaxo() || j - 3 < mesh.jmino() || j + 3 > mesh.jmaxo() ||
      k - 3 < mesh.kmino() || k + 3 > mesh.kmaxo())
    return -1;
  auto film = [&](const int a, const int b, const int c) {
    return film_is_gas ? 1.0 - vf(a, b, c) : vf(a, b, c);
  };
  const double f_centre = film(i, j, k);
  auto presence = [&](const int a, const int b, const int c) {
    double s = 0.0;
    for (int aa = a - 1; aa <= a + 1; ++aa)
      for (int bb = b - 1; bb <= b + 1; ++bb)
        for (int cc = c - 1; cc <= c + 1; ++cc) {
          const double f = film(aa, bb, cc);
          if (f > IRL::global_constants::VF_LOW) s += std::min(f / f_centre, 1.0);
        }
    return s;
  };

  // Orthonormal tangent basis (t1, t2): cross with the axis least aligned with the normal.
  const int least = std::abs(normal[0]) <= std::abs(normal[1]) && std::abs(normal[0]) <= std::abs(normal[2])
                        ? 0
                        : (std::abs(normal[1]) <= std::abs(normal[2]) ? 1 : 2);
  IRL::Normal axis(0.0, 0.0, 0.0);
  axis[least] = 1.0;
  IRL::Normal t1 = IRL::crossProduct(normal, axis);
  t1.normalize();
  const IRL::Normal t2 = IRL::crossProduct(normal, t1);

  int empty = 0;
  for (const int ring : {1, 2}) {
    // Unit direction to every cell of the ring (Chebyshev distance `ring`).
    std::array<std::array<int, 3>, 98> off;
    std::array<IRL::Normal, 98> dir;
    int n = 0;
    for (int a = -ring; a <= ring; ++a)
      for (int b = -ring; b <= ring; ++b)
        for (int c = -ring; c <= ring; ++c) {
          if (std::max({std::abs(a), std::abs(b), std::abs(c)}) != ring) continue;
          off[n] = {a, b, c};
          dir[n] = IRL::Normal(mesh.xm(i + a) - mesh.xm(i), mesh.ym(j + b) - mesh.ym(j), mesh.zm(k + c) - mesh.zm(k));
          dir[n].normalize();
          ++n;
        }
    // Best-aligned cell for each direction; each distinct cell is judged once.
    std::array<bool, 98> probed{};
    const double threshold = ring == 1 ? kInnerEmpty : kOuterEmpty;
    for (int m = 0; m < kDirections; ++m) {
      const double th = 2.0 * M_PI * m / kDirections;
      const IRL::Normal d = std::cos(th) * t1 + std::sin(th) * t2;
      int best = 0;
      for (int q = 1; q < n; ++q)
        if (d * dir[q] > d * dir[best]) best = q;
      if (probed[best]) continue;
      probed[best] = true;
      if (presence(i + off[best][0], j + off[best][1], k + off[best][2]) <= threshold) ++empty;
    }
  }
  return empty;
}

}  // namespace r2pedge

#endif  // EXAMPLES_NEW_ADVECTOR_R2P_EDGE_SENSOR_H_
