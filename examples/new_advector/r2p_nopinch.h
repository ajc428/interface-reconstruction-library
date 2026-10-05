// Pinch prevention for R2P-Net's two planes.
//
// Two planes placed by the Newton distance solve can meet inside the cell,
// giving the film zero thickness there. At a real edge that is the tip and is
// kept. Elsewhere (a neck: the faces converge, then level off into a thinner
// film) it pinches the film off, so the planes are opened less: both normals
// are rotated toward their mean normal m, each keeping its share of the
// opening,
//   n_i(l) = normalize((n_i.m) m + (1 - l) (n_i - (n_i.m) m)),
// with the distances re-solved, using the smallest l in [0, 1] (bisection) for
// which the film is nowhere in the cell thinner than kGapFraction times its
// mean thickness (l = 1: a parallel slab). Thicknesses are measured along m;
// the film's gap is linear in space, so its minimum over the cell is at a
// vertex of the polytope cell ∩ film, found exactly. The mean thickness is
// the film volume over the area of the plane through the film centroid along
// m, clipped to the cell: always positive, so planes
// that meet in the cell (gap 0, or ~1e-17 from rounding) never pass. (The gap
// at the film centroid, used before, is <= 0 when the centroid lies beyond
// where the planes meet, which let such planes through.)
//
// R2P_NOPINCH_GAP overrides kGapFraction for a run (< 0 turns this off).

#ifndef EXAMPLES_NEW_ADVECTOR_R2P_NOPINCH_H_
#define EXAMPLES_NEW_ADVECTOR_R2P_NOPINCH_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/r2p_edge_sensor.h"
#include "examples/new_advector/r2p_edge_topology.h"
#include "examples/new_advector/r2p_newton_distance.h"
#include "irl/generic_cutting/cut_polygon.h"
#include "irl/geometry/polygons/polygon.h"
#include "examples/new_advector/reconstruction_types.h"

namespace r2pnopinch {

inline double gapFraction() {
  static const double v = [] {
    const char* s = std::getenv("R2P_NOPINCH_GAP");
    return (s != nullptr && *s != '\0') ? std::strtod(s, nullptr) : 0.1;
  }();
  return v;
}

// The film as half-spaces N.x <= D (N out of the film: the liquid of an
// unflipped separator, the gas of a flipped one), and its mean normal m.
struct Film {
  IRL::Normal N[2], m;
  double D[2];
  explicit Film(const IRL::PlanarSeparator& sep) {
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
    double top = std::numeric_limits<double>::infinity(), bot = -top;
    for (int p = 0; p < 2; ++p) {
      const double nm = N[p] * m;
      if (std::abs(nm) < 1.0e-12) continue;
      const double s = (D[p] - N[p] * x) / nm;
      if (nm > 0.0) top = std::min(top, s);
      else bot = std::max(bot, s);
    }
    return top - bot;
  }
  // Thinnest film inside the box [lo, hi]: the gap at the vertices of box ∩ film
  // (0 on the line where the planes meet); +inf if the film misses the box.
  double minGap(const IRL::Pt& lo, const IRL::Pt& hi) const {
    IRL::Normal A[8];
    double B[8];
    for (int k = 0; k < 3; ++k) {
      A[2 * k] = IRL::Normal(k == 0, k == 1, k == 2);
      B[2 * k] = hi[k];
      A[2 * k + 1] = -A[2 * k];
      B[2 * k + 1] = -lo[k];
    }
    A[6] = N[0]; B[6] = D[0];
    A[7] = N[1]; B[7] = D[1];
    const double tol = 1.0e-10 * (hi[0] - lo[0]);
    double g = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 8; ++a)
      for (int b = a + 1; b < 8; ++b)
        for (int c = b + 1; c < 8; ++c) {
          const IRL::Normal bc = IRL::crossProduct(A[b], A[c]);
          const double det = A[a] * bc;
          if (std::abs(det) < 1.0e-12) continue;
          const IRL::Normal x = IRL::Normal(B[a] * bc + B[b] * IRL::crossProduct(A[c], A[a]) +
                                            B[c] * IRL::crossProduct(A[a], A[b])) / det;
          const IRL::Pt pt(x[0], x[1], x[2]);
          bool inside = true;
          for (int q = 0; q < 8 && inside; ++q) inside = A[q] * pt <= B[q] + tol;
          if (inside) g = std::min(g, gapAt(pt));
        }
    return g;
  }
};

// Opens the two planes of *sep (Newton-placed in the box [lo, hi]) until the
// film is nowhere thinner than gapFraction() times its mean thickness.
// Returns l (0 = unchanged).
inline double preventPinch(const IRL::Pt& lo, const IRL::Pt& hi, const double vf, const IRL::Pt& liq,
                           const IRL::Pt& gas, IRL::PlanarSeparator* sep) {
  if (gapFraction() < 0.0 || sep->getNumberOfPlanes() != 2) return 0.0;
  const bool flipped = sep->isFlipped();
  const IRL::Pt& centroid = flipped ? gas : liq;
  const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(lo, hi);
  const IRL::Normal m = Film(*sep).m, n0 = (*sep)[0].normal(), n1 = (*sep)[1].normal();
  // Mean film thickness: film volume over the area of the plane through the
  // film centroid along m, clipped to the cell (a full cell cross-section if
  // that plane misses the cell).
  const double L = hi[0] - lo[0], film_vf = flipped ? 1.0 - vf : vf, vol = cell.calculateVolume();
  double t_ref = film_vf * L;
  {
    const IRL::PlanarSeparator mid = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(m, m * centroid));
    const double area =
        std::abs(IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, mid, mid[0]).calculateVolume());
    if (area > 1.0e-12 * std::pow(vol, 2.0 / 3.0)) t_ref = film_vf * vol / area;
  }
  auto thick_enough = [&](const IRL::PlanarSeparator& s) {
    if (s.getNumberOfPlanes() != 2) return true;
    const double g = Film(s).minGap(lo, hi);
    return g >= gapFraction() * t_ref && g > 1.0e-12 * L;
  };
  if (thick_enough(*sep)) return 0.0;

  auto rotated = [&](const IRL::Normal& n, const double l) {
    const IRL::Normal along = (n * m) * m;
    IRL::Normal r = along + (1.0 - l) * (n - along);
    r.normalize();
    return r;
  };
  auto solve = [&](const double l) {
    IRL::PlanarSeparator s = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(rotated(n0, l), 0.0), IRL::Plane(rotated(n1, l), 0.0), flipped ? -1.0 : 1.0);
    r2pnewton::R2PNewtonDistanceSolver(vf, liq, gas, s, cell);
    return s;
  };
  double l_lo = 0.0, l_hi = 1.0;
  for (int it = 0; it < 14; ++it) {
    const double mid = 0.5 * (l_lo + l_hi);
    (thick_enough(solve(mid)) ? l_hi : l_lo) = mid;
  }
  *sep = solve(l_hi);
  return l_hi;
}

// Whether pinch prevention treats (i,j,k) as an edge (the planes may meet: the
// film ends): the topological sensor (r2p_edge_topology.h) unless
// R2P_EDGE_SENSOR=presence selects r2p_edge_sensor.h. The topological sensor
// is overruled where the thin-film guard holds: the guard found the other phase
// on the two sides of the film in separate regions without using a normal, so
// the film continues; the sensor's seeds, walked along R2P-Net's mean normal,
// can land on one side of a very thin film whose normals are poor and call it
// an edge. Fortran: r2p_is_edge.
// A thick film (edge_topo 2: no empty cell within 2 cells on a side) is
// skipped too: one cell's planes cannot pinch it off, and at a thick rounded rim
// opening the converging planes would push liquid past the real end.
inline bool isEdge(const int i, const int j, const int k) {
  if (!r2pedgetopo::useTopology()) return edge_sensor(i, j, k) >= r2pedge::kEdgeMinCount;
  if (edge_topo(i, j, k) >= 2.0) return true;
  return edge_topo(i, j, k) >= 1.0 && film_guard(i, j, k) == 0;
}

// The step for cell (i,j,k) of the field, skipped where isEdge. Records l in
// `unpinch`.
inline void applyAt(const BasicMesh& mesh, const int i, const int j, const int k, const double vf,
                    const IRL::Pt& liq, const IRL::Pt& gas, IRL::PlanarSeparator* sep) {
  if (isEdge(i, j, k)) return;
  const double l = preventPinch(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                                IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)), vf, liq, gas, sep);
  unpinch(i, j, k) = std::max(unpinch(i, j, k), l);
}

}  // namespace r2pnopinch

#endif  // EXAMPLES_NEW_ADVECTOR_R2P_NOPINCH_H_
