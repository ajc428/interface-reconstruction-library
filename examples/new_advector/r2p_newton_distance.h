// Two-plane distance solve matching volume fraction AND centroid.
//
// With the normals fixed, Newton solves for (d0, d1) so that the volume and
// the first moment across the film (along m = n0 - n1) match the cell. The
// moment target comes from the FILM phase's own centroid -- the gas when the
// separator is flipped -- so a thin gas film does not recover it from the
// liquid centroid, which would amplify any liquid/gas moment inconsistency by
// vf/(1 - vf).
//
// Jacobian from the interface polygons: moving plane i sweeps its polygon, so
// dV/dd_i = A_i and dM/dd_i = A_i p_i (liquid moments, both flip states).
// Volume is always conserved: the result is finished with a rigid shift of
// both planes that matches the volume fraction.

#ifndef R2P_NEWTON_DISTANCE_H_
#define R2P_NEWTON_DISTANCE_H_

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "irl/generic_cutting/cut_polygon.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/moments/volume_moments.h"
#include "irl/planar_reconstruction/planar_separator.h"

namespace r2pnewton {

// Rigid shift of both planes to match vf. IRL's solver first; it can miss on
// near-empty/near-full cells (VF ~ 1e-6), so bisect on the shift if it does.
// Liquid volume grows monotonically with the shift in both flip states.
inline void matchVolume(const IRL::RectangularCuboid& cell, const double vf,
                        IRL::PlanarSeparator& sep, const double vf_tol) {
  const double vol = cell.calculateVolume();
  auto error = [&](const IRL::PlanarSeparator& s) {
    return IRL::getVolumeMoments<IRL::VolumeMoments>(cell, s).volume() / vol - vf;
  };
  const IRL::PlanarSeparator start = sep;
  IRL::setDistanceToMatchVolumeFraction(cell, vf, &sep, vf_tol);
  if (std::abs(error(sep)) <= vf_tol) return;

  auto shifted = [&](const double t) {
    IRL::PlanarSeparator s = start;
    for (IRL::UnsignedIndex_t i = 0; i < s.getNumberOfPlanes(); ++i) s[i].distance() += t;
    return s;
  };
  double lo = -std::cbrt(vol), hi = -lo;
  while (error(shifted(lo)) > 0.0) lo *= 2.0;
  while (error(shifted(hi)) < 0.0) hi *= 2.0;
  for (int it = 0; it < 200 && hi - lo > DBL_EPSILON * (std::abs(lo) + std::abs(hi)); ++it) {
    const double mid = 0.5 * (lo + hi);
    sep = shifted(mid);
    const double e = error(sep);
    if (std::abs(e) <= vf_tol) return;
    (e < 0.0 ? lo : hi) = mid;
  }
}

inline void R2PNewtonDistanceSolver(double vf, IRL::Pt liquid_centroid,
                                    IRL::Pt gas_centroid, IRL::PlanarSeparator& sep,
                                    IRL::RectangularCuboid cell) {
  const double vf_tol = 1.0e-13;
  if (sep.getNumberOfPlanes() != 2) {
    IRL::setDistanceToMatchVolumeFraction(cell, vf, &sep, vf_tol);
    return;
  }

  const double vol = cell.calculateVolume(), L = std::cbrt(vol);
  const IRL::Pt cc = cell.calculateCentroid();
  IRL::Normal n0 = sep[0].normal(), n1 = sep[1].normal();
  n0.normalize();
  n1.normalize();
  IRL::Normal m = n0 - n1;  // across the film
  if (m.calculateMagnitude() < 1.0e-12) m = n0;
  m.normalize();

  // Initial guess: both planes through the film centroid, shifted rigidly to
  // match volume.
  const bool flipped = sep.isFlipped();
  const IRL::Pt& film = flipped ? gas_centroid : liquid_centroid;
  sep[0] = IRL::Plane(n0, n0 * film);
  sep[1] = IRL::Plane(n1, n1 * film);
  matchVolume(cell, vf, sep, vf_tol);

  // Newton on (d0, d1) for r = (V/vol - vf, m.(M - M_target)/(vol L)), with M
  // the liquid first moment; flipped, its target is built from the gas side.
  const double m_target =
      flipped ? m * cc - (1.0 - vf) * (m * gas_centroid) : vf * (m * liquid_centroid);
  IRL::PlanarSeparator best = sep;
  double best_merit = DBL_MAX;
  for (int it = 0; it < 20; ++it) {
    const auto moments = IRL::getVolumeMoments<IRL::VolumeMoments>(cell, sep);
    const double r0 = moments.volume() / vol - vf;
    const double r1 = (m * moments.centroid() / vol - m_target) / L;
    const double merit = r0 * r0 + r1 * r1;
    if (merit < best_merit) {
      best_merit = merit;
      best = sep;
    }
    if (std::abs(r0) < vf_tol && std::abs(r1) < vf_tol) break;

    double A[2], Ap[2];  // polygon area and m-first-moment, per plane
    for (int i = 0; i < 2; ++i) {
      const auto pm = IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, sep, sep[i])
                          .calculateMoments();
      const double sign = pm.volume() < 0.0 ? -1.0 : 1.0;
      A[i] = sign * pm.volume() / vol;
      Ap[i] = sign * (m * pm.centroid()) / (vol * L);
    }
    const double det = A[0] * Ap[1] - A[1] * Ap[0];
    if (std::abs(det) < 1.0e-14) break;  // one plane has left the cell
    double dd0 = -(Ap[1] * r0 - A[1] * r1) / det;
    double dd1 = -(A[0] * r1 - Ap[0] * r0) / det;
    // Residuals are only piecewise smooth: keep a step within one cell size.
    const double scale = std::max(1.0, std::max(std::abs(dd0), std::abs(dd1)) / L);
    sep[0].distance() += dd0 / scale;
    sep[1].distance() += dd1 / scale;
  }

  sep = best;
  matchVolume(cell, vf, sep, vf_tol);
  IRL::cleanReconstruction(cell, vf, &sep);
}

// Liquid centroid only (R2PDistanceSolver's signature): the gas centroid is
// taken as the one consistent with it.
inline void R2PNewtonDistanceSolver(double vf, IRL::Pt liquid_centroid,
                                    IRL::PlanarSeparator& sep, IRL::RectangularCuboid cell) {
  const IRL::Pt cc = cell.calculateCentroid();
  IRL::Pt gas = cc;
  if (1.0 - vf > 1.0e-8)
    for (int d = 0; d < 3; ++d) gas[d] = (cc[d] - vf * liquid_centroid[d]) / (1.0 - vf);
  R2PNewtonDistanceSolver(vf, liquid_centroid, gas, sep, cell);
}

}  // namespace r2pnewton

#endif  // R2P_NEWTON_DISTANCE_H_
