// Two-plane distance solve matching volume fraction AND centroid.
//
// With the normals fixed, a two-plane reconstruction has two free distances.
// The cell supplies two usable constraints: the liquid volume fraction, and
// the component of the liquid centroid along the direction across the film
// (at fixed normals the distances cannot move the other two components). So
// the system is exactly determined.
//
// The deployed R2PDistanceSolver instead fixes the split from the centroid
// projection and bisects a single common shift for volume alone, which
// assumes the film's mid-plane sits at its centroid. That holds for a thin
// film crossing the whole cell, and fails once the cell's corners clip a
// thick one -- measured at 32% of flat slabs 0.3-0.6 cells thick landing
// above 5% symmetric-difference error even with exact normals.
//
// Parameterization (c_i = n_i . x_cell, so s = g = 0 puts both planes
// through the cell centre):
//
//   d0 = c0 + s + g,   d1 = c1 + s - g
//
// For a slab (n1 = -n0) the liquid is n0.x in [c0 - s + g, c0 + s + g]: s is
// the half-thickness and g the position along the film normal. Volume is
// monotone in s in both flip states (the intersection and the union of two
// half-spaces both grow as the half-spaces grow), and sliding g moves the
// centroid across the film, so each solve is a monotone root find.

#ifndef EXAMPLES_R2PNET_BENCH_DISTANCE_SOLVE_H_
#define EXAMPLES_R2PNET_BENCH_DISTANCE_SOLVE_H_

#include <cmath>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/plane_distance.h"
#include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
#include "irl/moments/volume_moments.h"
#include "irl/planar_reconstruction/planar_separator.h"

namespace bench {

// Cell cuts spent, for cost comparison against the Newton solver.
inline long g_bracketed_cuts = 0;
inline long g_bracketed_solves = 0;

// Illinois (modified regula falsi) on a bracketed monotone function.
// Superlinear where bisection is linear, which matters because every
// evaluation is an IRL cell cut: this turns a ~2000-cut nested bisection
// into ~150 cuts. It keeps a valid bracket throughout, so it cannot run
// away like a plain secant iteration.
template <class F>
inline double solveBracketed(F&& f, double lo, double hi, double flo, double fhi,
                             double ftol = 1.0e-12, double xtol = 1.0e-12, int maxit = 60) {
  if (flo == 0.0) return lo;
  if (fhi == 0.0) return hi;
  if (flo * fhi > 0.0) return std::abs(flo) < std::abs(fhi) ? lo : hi;   // no sign change
  double x = 0.5 * (lo + hi);
  for (int it = 0; it < maxit; ++it) {
    const double denom = fhi - flo;
    x = (std::abs(denom) > 1.0e-300) ? (lo * fhi - hi * flo) / denom : 0.5 * (lo + hi);
    // Keep the trial point strictly inside; regula falsi can otherwise
    // creep onto an endpoint and stall.
    const double margin = 1.0e-3 * (hi - lo);
    x = std::min(hi - margin, std::max(lo + margin, x));
    const double fx = f(x);
    if (std::abs(fx) <= ftol || (hi - lo) <= xtol) return x;
    if ((fx < 0.0) == (flo < 0.0)) {
      lo = x;
      flo = fx;
      fhi *= 0.5;   // Illinois: halve the stale endpoint's weight
    } else {
      hi = x;
      fhi = fx;
      flo *= 0.5;
    }
  }
  return x;
}

// Solves both distances for fixed normals. `liq` is the target LIQUID
// centroid regardless of flip state (IRL reports liquid moments either way,
// so no phase conversion is needed here).
inline IRL::PlanarSeparator solveTwoDistances(const IRL::Normal& n0, const IRL::Normal& n1,
                                              double flip, double vf, const IRL::Pt& liq,
                                              const IRL::RectangularCuboid& cell) {
  ++g_bracketed_solves;
  const IRL::Pt cc = cell.calculateCentroid();
  const double c0 = n0 * cc, c1 = n1 * cc;
  const double vol = cell.calculateVolume();

  IRL::Normal m = n0 - n1;                    // across the film; ~2*n0 for a slab
  if (m.calculateMagnitude() < 1.0e-12) m = n0;
  m.normalize();

  auto build = [&](double s, double g) {
    return IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(n0, c0 + s + g),
                                               IRL::Plane(n1, c1 + s - g), flip);
  };
  auto moments = [&](const IRL::PlanarSeparator& sep) {
    ++g_bracketed_cuts;
    return IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep);
  };

  // s such that the volume fraction is matched, for a given g.
  const double kS = 3.0;   // s = +/-3 cell widths brackets empty .. full
  auto solveS = [&](double g) {
    auto fs = [&](double s) { return moments(build(s, g)).volume() / vol - vf; };
    return solveBracketed(fs, -kS, kS, fs(-kS), fs(kS), 1.0e-13, 1.0e-13);
  };

  // g such that the centroid component across the film is matched, with the
  // volume re-matched at every trial so no candidate violates it.
  const double target = m * liq;
  auto resid = [&](double g) {
    const auto mom = moments(build(solveS(g), g));
    return (mom.volume() > 1.0e-14 ? m * mom.centroid() : m * cc) - target;
  };
  const double kG = 1.5;   // spans the cell plus margin
  const double g = solveBracketed(resid, -kG, kG, resid(-kG), resid(kG), 1.0e-11, 1.0e-11);

  IRL::PlanarSeparator sep = build(solveS(g), g);
  IRL::cleanReconstruction(cell, vf, &sep);
  return sep;
}

// Drop-in replacement for R2PDistanceSolver: same arguments, same in-place
// update, but solves both distances. Single-plane separators keep the
// ordinary volume-conserving translation.
inline void momentDistanceSolver(double vf, const IRL::Pt& liq, IRL::PlanarSeparator& sep,
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
  sep = solveTwoDistances(n0, n1, sep.isNotFlipped() ? 1.0 : -1.0, vf, liq, cell);
}

}  // namespace bench

#endif  // EXAMPLES_R2PNET_BENCH_DISTANCE_SOLVE_H_
