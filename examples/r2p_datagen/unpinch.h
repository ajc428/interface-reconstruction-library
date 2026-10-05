// Label constraint: two-plane labels that do not pinch a continuing film.
//
// The legacy labels are the centre cell's area-averaged face normals. Where a
// film's faces curve (a neck: converging, then levelling off), two planes with
// those normals keep converging past where the real faces level off and can
// cross inside the cell, although the real film never gets thinner than its
// neck there. Deployed, that pinches the film off.
//
// For nested scenes (sheet, wedge, neck: faces verified apart across the
// stencil), the labels are placed as R2P3D_Net does -- two planes, distances
// from the Newton solve on the centre cell's VF and centroids as the network
// sees them (after the generator's centroid noise) -- and the
// thinnest film of those planes inside the centre cell is compared with the
// thinnest true film there, both measured along the mean film normal m. The
// planes' gap is linear in space, so its minimum over their film inside the
// cell (the cube cut by both planes) is at a vertex of that polytope, found
// exactly; planes that meet inside the cell have a vertex on that line, gap 0.
// The true film is sampled over columns of the exact faces that cross the
// cell. If the planes are thinner than
// min_gap_fraction times the true minimum, both normals are rotated toward m,
// each keeping its share of the opening,
//   n_i(l) = normalize((n_i.m) m + (1 - l) (n_i - (n_i.m) m)),
// with the smallest l in [0, 1] (bisection) that restores the bound. Real tips
// (edges, tongues) are not nested scenes and are left alone.

#ifndef EXAMPLES_R2P_DATAGEN_UNPINCH_H_
#define EXAMPLES_R2P_DATAGEN_UNPINCH_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "examples/new_advector/r2p_newton_distance.h"
#include "examples/r2p_datagen/stencil.h"

namespace r2pgen {

// Roots of A s^2 + B s + C = 0 in [lo, hi]; returns the count (0-2).
inline int quadRoots(const double A, const double B, const double C, const double lo, const double hi,
                     double* r) {
  int n = 0;
  auto keep = [&](const double s) { if (s >= lo && s <= hi) r[n++] = s; };
  if (std::abs(A) < 1.0e-14) {
    if (std::abs(B) > 1.0e-14) keep(-C / B);
  } else {
    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return 0;
    const double q = -0.5 * (B + std::copysign(std::sqrt(disc), B));
    double s0 = q / A, s1 = (q != 0.0) ? C / q : s0;
    if (s0 > s1) std::swap(s0, s1);
    keep(s0);
    if (s1 != s0) keep(s1);
  }
  return n;
}

// Parameter where the line c + s m crosses paraboloid p, nearest to s = 0.
inline bool crossing(const Para& p, const Vec3& c, const Vec3& m, double* s) {
  const Vec3 r = c - p.d;
  const double p0 = r.dot(p.e0), p1 = r.dot(p.e1), p2 = r.dot(p.e2);
  const double q0 = m.dot(p.e0), q1 = m.dot(p.e1), q2 = m.dot(p.e2);
  const double A = p.a * q0 * q0 + p.b * q1 * q1;
  const double B = q2 + 2.0 * p.a * p0 * q0 + 2.0 * p.b * p1 * q1;
  const double C = p2 + p.a * p0 * p0 + p.b * p1 * p1;
  double roots[2];
  const int n = quadRoots(A, B, C, -4.0, 4.0, roots);
  if (n == 0) return false;
  *s = (n == 2 && std::abs(roots[1]) < std::abs(roots[0])) ? roots[1] : roots[0];
  return true;
}

// Does the segment c + s m, s in [s0, s1], meet the centre cell [-1/2, 1/2]^3?
inline bool segmentInCell(const Vec3& c, const Vec3& m, double s0, double s1) {
  for (int k = 0; k < 3; ++k) {
    if (std::abs(m[k]) < 1.0e-14) {
      if (c[k] < -0.5 || c[k] > 0.5) return false;
      continue;
    }
    double a = (-0.5 - c[k]) / m[k], b = (0.5 - c[k]) / m[k];
    if (a > b) std::swap(a, b);
    s0 = std::max(s0, a);
    s1 = std::min(s1, b);
    if (s0 > s1) return false;
  }
  return true;
}

// Thinnest film of two planes (liquid: n.x <= d for both) inside the centre
// cell, measured along m: the minimum of the (linear) gap over the vertices of
// cell ∩ both half-spaces. +inf if they leave no liquid in the cell.
inline double planeFilmMin(const Vec3 n[2], const double d[2], const Vec3& m) {
  Vec3 N[8];
  double D[8];
  for (int k = 0; k < 3; ++k) {
    N[2 * k] = Vec3::Unit(k); D[2 * k] = 0.5;
    N[2 * k + 1] = -Vec3::Unit(k); D[2 * k + 1] = 0.5;
  }
  N[6] = n[0]; D[6] = d[0];
  N[7] = n[1]; D[7] = d[1];
  auto gap = [&](const Vec3& x) {
    double top = std::numeric_limits<double>::infinity(), bot = -std::numeric_limits<double>::infinity();
    for (int p = 0; p < 2; ++p) {
      const double nm = n[p].dot(m);
      if (std::abs(nm) < 1.0e-12) continue;
      const double s = (d[p] - n[p].dot(x)) / nm;
      if (nm > 0.0) top = std::min(top, s);
      else bot = std::max(bot, s);
    }
    return top - bot;
  };
  double g = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 8; ++i)
    for (int j = i + 1; j < 8; ++j)
      for (int k = j + 1; k < 8; ++k) {
        Mat3 A;
        A.row(0) = N[i]; A.row(1) = N[j]; A.row(2) = N[k];
        if (std::abs(A.determinant()) < 1.0e-12) continue;
        const Vec3 x = A.partialPivLu().solve(Vec3(D[i], D[j], D[k]));
        bool inside = true;
        for (int q = 0; q < 8 && inside; ++q) inside = N[q].dot(x) <= D[q] + 1.0e-10;
        if (inside) g = std::min(g, gap(x));
      }
  return g;
}

// Rotates the two face normals (unit, out of the liquid) of a nested scene so
// their Newton-placed planes are no thinner inside the centre cell than
// min_gap_fraction times the true film. centre: the centre cell's 7 moments
// as the network sees them (noisy). Returns the rotation parameter l (0 =
// labels unchanged).
inline double unpinchLabels(const Scene& sc, const double* centre, const double min_gap_fraction,
                            Faces* faces) {
  if (sc.kind != Kind::kNested || faces->area[0] <= 0.0 || faces->area[1] <= 0.0) return 0.0;
  const Vec3 n_lo = faces->normal[0], n_up = faces->normal[1];   // lower face points down, upper up
  Vec3 m = n_up - n_lo;
  if (m.norm() < 1.0e-12) return 0.0;
  m.normalize();
  const Vec3 t1 = (std::abs(m.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY()).cross(m).normalized();
  const Vec3 t2 = m.cross(t1);

  // True film: thinnest over the columns along m where it crosses the cell.
  constexpr int kG = 25;
  const double reach = 0.8660254037844386;   // cell half-diagonal
  int ncol = 0;
  double true_min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < kG; ++i)
    for (int j = 0; j < kG; ++j) {
      const Vec3 c = (-reach + 2.0 * reach * i / (kG - 1)) * t1 + (-reach + 2.0 * reach * j / (kG - 1)) * t2;
      double s_up, s_lo;
      if (!crossing(sc.upper, c, m, &s_up) || !crossing(sc.lower, c, m, &s_lo) || s_up <= s_lo) continue;
      if (!segmentInCell(c, m, s_lo, s_up)) continue;
      ++ncol;
      true_min = std::min(true_min, s_up - s_lo);
    }
  if (ncol == 0 || !(true_min > 0.0)) return 0.0;
  const double target = min_gap_fraction * true_min;

  const IRL::RectangularCuboid cell = unitCell(Vec3::Zero());
  const IRL::Pt liq(centre[1], centre[2], centre[3]), gas(centre[4], centre[5], centre[6]);
  auto rotated = [&](const Vec3& n, const double l) {
    const Vec3 along = n.dot(m) * m;
    return Vec3((along + (1.0 - l) * (n - along)).normalized());
  };
  // Thinnest film of the Newton-placed planes in the cell; +inf if the solve
  // keeps fewer than two planes.
  auto planeGap = [&](const double l) {
    const Vec3 a = rotated(n_lo, l), b = rotated(n_up, l);
    IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(IRL::Normal(a.x(), a.y(), a.z()), 0.0), IRL::Plane(IRL::Normal(b.x(), b.y(), b.z()), 0.0), 1.0);
    r2pnewton::R2PNewtonDistanceSolver(centre[0], liq, gas, sep, cell);
    if (sep.getNumberOfPlanes() != 2) return std::numeric_limits<double>::infinity();
    const Vec3 n[2] = {Vec3(sep[0].normal()[0], sep[0].normal()[1], sep[0].normal()[2]),
                       Vec3(sep[1].normal()[0], sep[1].normal()[1], sep[1].normal()[2])};
    const double d[2] = {sep[0].distance(), sep[1].distance()};
    return planeFilmMin(n, d, m);
  };
  if (planeGap(0.0) >= target) return 0.0;
  double lo = 0.0, hi = 1.0;
  if (planeGap(hi) < target) {
    lo = hi;   // even a slab is too thin here: use it
  } else {
    for (int it = 0; it < 14; ++it) {
      const double mid = 0.5 * (lo + hi);
      (planeGap(mid) >= target ? hi : lo) = mid;
    }
    lo = hi;
  }
  faces->normal[0] = rotated(n_lo, lo);
  faces->normal[1] = rotated(n_up, lo);
  return lo;
}

}  // namespace r2pgen

#endif  // EXAMPLES_R2P_DATAGEN_UNPINCH_H_
