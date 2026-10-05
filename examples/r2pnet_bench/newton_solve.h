// Newton version of the two-plane distance solve.
//
// Same problem as distance_solve.h -- fixed normals, find the two distances
// that match the liquid volume fraction and the liquid centroid component
// across the film -- but solved with exact derivatives instead of nested
// bracketing. This is the form intended for production; the bracketed solver
// stays as the reference it is validated against.
//
// WHY IT IS CHEAP. Moving plane i outward by a small delta sweeps a slab of
// thickness delta over that plane's polygon inside the cell, so
//
//     dV/dd_i = A_i            A_i = area of plane i's polygon
//     dM/dd_i = A_i * p_i      p_i = that polygon's centroid
//
// with M the liquid first moment. Both A_i and p_i come out of the same cut
// that produced V and M, so a Newton step costs one cut rather than the
// ~150 the bracketed solver spends. The polygon is the part of the plane
// that actually bounds the liquid: for an unflipped separator (liquid =
// intersection of the half-spaces) that is the piece inside the other
// half-space, and for a flipped one (liquid = union) it is the piece
// outside it. Either way dV/dd_i = +A_i.
//
// VARIABLES. Same (s, g) as distance_solve.h:
//
//     d0 = c0 + s + g,   d1 = c1 + s - g,   c_i = n_i . x_cell
//
// For a slab s is the half-thickness and g the position along the film
// normal, which keeps the Jacobian near-diagonal (s moves volume at nearly
// fixed centroid, g moves the centroid at nearly fixed volume) even when the
// two normals are almost antiparallel and the raw (d0, d1) system is not.
//
// NON-SMOOTHNESS. The residuals are only piecewise smooth: derivatives jump
// when a plane leaves the cell or the planes' intersection line crosses it.
// Newton is therefore damped, watched, and backed by the bracketed solver,
// which is what runs if it fails to converge.

#ifndef EXAMPLES_R2PNET_BENCH_NEWTON_SOLVE_H_
#define EXAMPLES_R2PNET_BENCH_NEWTON_SOLVE_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/plane_distance.h"
#include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
#include "irl/moments/volume_moments.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "irl/machine_learning_reconstruction/plic_geometry.h"

#include "examples/new_advector/r2p_paraboloid_pass.h"   // r2pgeom helpers
#include "examples/r2pnet_bench/distance_solve.h"        // bracketed fallback

namespace bench {

// Cell cuts performed, so the benchmark can report cost in a unit that does
// not depend on the machine.
inline long g_newton_cuts = 0;
inline long g_newton_solves = 0;
inline long g_newton_fallbacks = 0;

struct CutState {
  double vf = 0.0;             // liquid volume fraction
  IRL::Pt centroid;            // liquid centroid
  double area[2] = {0.0, 0.0}; // bounding polygon area of each plane
  IRL::Pt poly_centroid[2];    // and its centroid
  double volume = 0.0;         // liquid volume
};

// One cut: volume, centroid, and both bounding polygons.
inline CutState evaluateCut(const IRL::PlanarSeparator& sep, const IRL::RectangularCuboid& cell,
                            const IRL::Pt& lo, const IRL::Pt& hi) {
  ++g_newton_cuts;
  CutState st;
  const double vol = cell.calculateVolume();
  const auto mom = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep);
  st.volume = mom.volume();
  st.vf = st.volume / vol;
  st.centroid = st.volume > 1.0e-14 * vol ? mom.centroid() : cell.calculateCentroid();

  const bool flipped = sep.isFlipped();
  for (int p = 0; p < 2; ++p) {
    const auto info = plicgeom::planeBoxPolygon(lo, hi, sep[p].normal(), sep[p].distance());
    if (!info) continue;
    std::vector<IRL::Pt> verts = info->vertices;
    const int q = 1 - p;
    // Keep the part of plane p that bounds the liquid: inside the other
    // half-space when the liquid is their intersection, outside it when the
    // liquid is their union.
    if (!flipped) {
      verts = r2pgeom::clipToHalfSpace(verts, sep[q].normal(), sep[q].distance());
    } else {
      IRL::Normal opposite = sep[q].normal();
      opposite = -opposite;
      verts = r2pgeom::clipToHalfSpace(verts, opposite, -sep[q].distance());
    }
    if (verts.size() < 3) continue;
    double area = 0.0;
    IRL::Pt c;
    if (!r2pgeom::polygonAreaCentroid(verts, &area, &c)) continue;
    st.area[p] = area;
    st.poly_centroid[p] = c;
  }
  return st;
}

// Why a Newton solve gave up, so the fallback rate can be attributed rather
// than guessed at.
enum class NewtonFail {
  kNone = 0,
  kNoArea,      // neither plane cuts the cell any more
  kTinyVolume,  // liquid volume ~ 0, so the centroid row is meaningless
  kSingular,    // Jacobian determinant below tolerance
  kStalled,     // backtracking could not improve the merit function
  kMaxIter,     // ran out of iterations while still improving
  kOnePlaneOnly // only one face bounds the liquid: volume is all there is
};
inline long g_newton_fail_reason[7] = {0, 0, 0, 0, 0, 0, 0};
inline long g_newton_one_plane = 0;

struct NewtonResult {
  IRL::PlanarSeparator separator;
  bool converged = false;
  int iterations = 0;
  NewtonFail reason = NewtonFail::kNone;
  double residual_vf = 0.0;
  double residual_cm = 0.0;
};

// Newton solve in (s, g). Returns converged = false if it stalls, leaves the
// cell, or hits a singular Jacobian; the caller then uses the bracketed
// solver.
inline NewtonResult solveTwoDistancesNewton(const IRL::Normal& n0, const IRL::Normal& n1,
                                            double flip, double vf, const IRL::Pt& liq,
                                            const IRL::RectangularCuboid& cell,
                                            double s_init, double g_init,
                                            double ftol = 1.0e-13, int maxit = 20) {
  const IRL::Pt cc = cell.calculateCentroid();
  const double c0 = n0 * cc, c1 = n1 * cc;
  const double vol = cell.calculateVolume();
  // Corner points from the centroid and side lengths, rather than relying on
  // the vertex ordering of the stored cuboid.
  const IRL::Pt half(0.5 * cell.calculateSideLength(0), 0.5 * cell.calculateSideLength(1),
                     0.5 * cell.calculateSideLength(2));
  const IRL::Pt lo(cc[0] - half[0], cc[1] - half[1], cc[2] - half[2]);
  const IRL::Pt hi(cc[0] + half[0], cc[1] + half[1], cc[2] + half[2]);

  IRL::Normal m = n0 - n1;
  if (m.calculateMagnitude() < 1.0e-12) m = n0;
  m.normalize();
  // Pt differences in IRL are expression templates, so the projections are
  // written out componentwise rather than with operator*.
  auto proj = [&m](const IRL::Pt& a, const IRL::Pt& b) {
    return m[0] * (a[0] - b[0]) + m[1] * (a[1] - b[1]) + m[2] * (a[2] - b[2]);
  };
  const double target = m * liq;

  auto build = [&](double s, double g) {
    return IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(n0, c0 + s + g),
                                               IRL::Plane(n1, c1 + s - g), flip);
  };

  // Volume-only solve in s at fixed g. Used for the warm start, and again
  // when only one face bounds the liquid.
  auto solveVolume = [&](double g_fixed, double ftol_v) {
    auto fs = [&](double s_try) {
      ++g_newton_cuts;
      return IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, build(s_try, g_fixed)).volume() / vol - vf;
    };
    const double kS = 3.0;
    // The endpoints are known without cutting: at s = -3 the region is empty
    // and at s = +3 it fills the cell, for unit normals in a unit cell.
    return solveBracketed(fs, -kS, kS, -vf, 1.0 - vf, ftol_v, 1.0e-12, 30);
  };

  NewtonResult out;
  double s = s_init, g = g_init;

  // Warm start on s alone. Starting at s = 0 puts both planes through the
  // cell centre, where a parallel film encloses zero volume: A0 = A1, the
  // two Jacobian rows become degenerate and Newton has nothing to work
  // with. A loose 1-D solve for the volume first lands in a non-degenerate
  // state, after which Newton converges in a couple of steps.
  s = solveVolume(g, 1.0e-6);

  IRL::PlanarSeparator sep = build(s, g);
  CutState st = evaluateCut(sep, cell, lo, hi);
  double f0 = st.vf - vf;
  double f1 = (m * st.centroid) - target;
  double merit = f0 * f0 + f1 * f1;

  for (int it = 0; it < maxit; ++it) {
    out.iterations = it;
    if (std::abs(f0) <= ftol && std::abs(f1) <= ftol) {
      out.converged = true;
      break;
    }

    // Jacobian of (vf, m.centroid) with respect to (s, g).
    //   dd0/ds = dd1/ds = 1,  dd0/dg = +1, dd1/dg = -1
    //   d(vf)/dd_i   = A_i / vol
    //   d(m.C)/dd_i  = A_i * m.(p_i - C) / V
    const double A0 = st.area[0], A1 = st.area[1];
    if (A0 + A1 <= 1.0e-12) { out.reason = NewtonFail::kNoArea; break; }
    if (st.volume <= 1.0e-9 * vol) { out.reason = NewtonFail::kTinyVolume; break; }
    const double V = st.volume;
    const double a0 = A0 * proj(st.poly_centroid[0], st.centroid) / V;
    const double a1 = A1 * proj(st.poly_centroid[1], st.centroid) / V;

    const double J[2][2] = {{(A0 + A1) / vol, (A0 - A1) / vol}, {a0 + a1, a0 - a1}};
    const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    // det = 2 A0 A1 (q0 - q1) / (V vol), so it vanishes when one face has no
    // bounding polygon -- the film is thicker than the cell and only one
    // plane cuts it. The centroid across the film is then not reachable by
    // moving planes at all, so matching the volume is the whole problem:
    // solve that in 1-D and accept, rather than paying for a 2-D fallback
    // that can do no better.
    // det = 2 A0 A1 (q0 - q1) / (V vol), so it vanishes when one face has no
    // bounding polygon: the film is thicker than the cell, or tilted out of
    // it, and only one plane bounds the liquid. The liquid region is then
    // exactly one half-space, which IRL solves directly and exactly -- far
    // better than a 2-D fallback that has no second degree of freedom to
    // use. The other plane is left where it is; if the new position brings
    // it back into contact, the loop simply carries on.
    if (std::abs(det) <= 1.0e-14) {
      const bool zero0 = A0 <= 1.0e-12, zero1 = A1 <= 1.0e-12;
      if (zero0 != zero1) {
        ++g_newton_one_plane;
        const IRL::Normal& n_act = zero1 ? n0 : n1;
        const double d_act = IRL::findDistanceOnePlane(cell, vf, n_act);
        ++g_newton_cuts;
        const double d_other = zero1 ? sep[1].distance() : sep[0].distance();
        sep = zero1 ? IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(n0, d_act),
                                                          IRL::Plane(n1, d_other), flip)
                    : IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(n0, d_other),
                                                          IRL::Plane(n1, d_act), flip);
        // Recover (s, g) from the distances so the loop can continue.
        const double e0 = sep[0].distance() - c0, e1 = sep[1].distance() - c1;
        s = 0.5 * (e0 + e1);
        g = 0.5 * (e0 - e1);
        st = evaluateCut(sep, cell, lo, hi);
        f0 = st.vf - vf;
        f1 = (m * st.centroid) - target;
        merit = f0 * f0 + f1 * f1;
        if (st.area[0] > 1.0e-12 && st.area[1] > 1.0e-12) continue;   // back in contact
        // Still one-sided: the centroid across the film is unreachable, and
        // matching the volume is the whole of the problem.
        out.reason = NewtonFail::kOnePlaneOnly;
        out.converged = std::abs(f0) <= 1.0e-12;
        out.residual_vf = std::abs(f0);
        out.residual_cm = std::abs(f1);
        out.separator = sep;
        if (!out.converged) ++g_newton_fail_reason[static_cast<int>(out.reason)];
        return out;
      }
      out.reason = NewtonFail::kSingular;
      break;
    }

    const double ds = (-f0 * J[1][1] + f1 * J[0][1]) / det;
    const double dg = (-J[0][0] * f1 + J[1][0] * f0) / det;

    // Damped step: the residuals are piecewise smooth, so a full Newton step
    // can jump past a kink. Backtrack until the merit function improves.
    bool improved = false;
    double lambda = 1.0;
    for (int back = 0; back < 8; ++back) {
      const double s_try = s + lambda * ds, g_try = g + lambda * dg;
      const IRL::PlanarSeparator sep_try = build(s_try, g_try);
      const CutState st_try = evaluateCut(sep_try, cell, lo, hi);
      const double f0_try = st_try.vf - vf;
      const double f1_try = (m * st_try.centroid) - target;
      const double merit_try = f0_try * f0_try + f1_try * f1_try;
      if (merit_try < merit) {
        s = s_try; g = g_try; sep = sep_try; st = st_try;
        f0 = f0_try; f1 = f1_try; merit = merit_try;
        improved = true;
        break;
      }
      lambda *= 0.5;
    }
    if (!improved) { out.reason = NewtonFail::kStalled; break; }   // hand over to the bracketed solver
    if (it + 1 == maxit) out.reason = NewtonFail::kMaxIter;
  }

  if (std::abs(f0) <= ftol && std::abs(f1) <= ftol) {
    out.converged = true;
    out.reason = NewtonFail::kNone;
  }
  // Residuals when it gave up, for attributing the fallbacks.
  out.residual_vf = std::abs(f0);
  out.residual_cm = std::abs(f1);
  out.separator = sep;
  if (!out.converged) ++g_newton_fail_reason[static_cast<int>(out.reason)];
  return out;
}

// Drop-in R2PDistanceSolver replacement: Newton first, bracketed solver as
// the fallback. The warm start reuses the deployed heuristic's split, which
// is the projection of the target centroid across the film.
inline void newtonDistanceSolver(double vf, const IRL::Pt& liq, IRL::PlanarSeparator& sep,
                                 const IRL::RectangularCuboid& cell) {
  if (sep.getNumberOfPlanes() == 0) return;
  if (sep.getNumberOfPlanes() == 1) {
    IRL::Normal n = sep[0].normal();
    if (n.calculateMagnitude() < 1.0e-12) return;
    n.normalize();
    sep = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, IRL::findDistanceOnePlane(cell, vf, n)));
    return;
  }
  IRL::Normal n0 = sep[0].normal(), n1 = sep[1].normal();
  if (n0.calculateMagnitude() < 1.0e-12 || n1.calculateMagnitude() < 1.0e-12) return;
  n0.normalize();
  n1.normalize();

  ++g_newton_solves;
  const double flip = sep.isNotFlipped() ? 1.0 : -1.0;
  const IRL::Pt cc = cell.calculateCentroid();
  IRL::Normal m = n0 - n1;
  if (m.calculateMagnitude() < 1.0e-12) m = n0;
  m.normalize();
  const double g0 = m[0] * (liq[0] - cc[0]) + m[1] * (liq[1] - cc[1]) + m[2] * (liq[2] - cc[2]);

  const NewtonResult res =
      solveTwoDistancesNewton(n0, n1, flip, vf, liq, cell, 0.0, g0);
  if (res.converged) {
    sep = res.separator;
    IRL::cleanReconstruction(cell, vf, &sep);
    return;
  }
  ++g_newton_fallbacks;
  sep = solveTwoDistances(n0, n1, flip, vf, liq, cell);
}

}  // namespace bench

#endif  // EXAMPLES_R2PNET_BENCH_NEWTON_SOLVE_H_
