// Centre-cell error metrics.
//
// A PlanarSeparator is turned into column intervals exactly like the truth
// region, with each endpoint tagged by the plane that produced it. That gives
// the symmetric-difference volume and, per plane, the patch of the
// reconstructed interface that actually bounds the liquid (a plane clipped
// away by its partner contributes nothing) -- measured with the same
// orientation rule as the truth, so no sign conventions can leak in.

#ifndef EXAMPLES_R2PNET_BENCH_METRICS_H_
#define EXAMPLES_R2PNET_BENCH_METRICS_H_

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/r2pnet_bench/geometry.h"

namespace bench {

// How IRL interprets a flipped separator. Determined at start-up by
// calibrateFlipConvention() rather than assumed.
//   kUnionBelow:        liquid = union_i { n_i.x < d_i }
//   kComplementOfBelow: liquid = complement of intersection_i { n_i.x <= d_i }
enum class FlipConvention { kUnionBelow, kComplementOfBelow };
inline FlipConvention g_flip_convention = FlipConvention::kUnionBelow;

inline void planeBelowInterval(const IRL::Plane& p, int tag, const Vec3& base, int axis,
                               double t0, double t1, IntervalList* out) {
  Quadric q;
  q.b = {p.normal()[0], p.normal()[1], p.normal()[2]};
  q.c = -p.distance();
  quadricIntervals(q, tag, base, axis, t0, t1, out);
}

inline void separatorColumn(const IRL::PlanarSeparator& sep, const Vec3& base, int axis,
                            double t0, double t1, IntervalList* out) {
  const int np = static_cast<int>(sep.getNumberOfPlanes());
  IntervalList acc, cur, tmp;
  const bool flipped = sep.isFlipped();
  const bool use_union = flipped && g_flip_convention == FlipConvention::kUnionBelow;
  for (int p = 0; p < np; ++p) {
    planeBelowInterval(sep[p], p, base, axis, t0, t1, &cur);
    if (p == 0) { acc = cur; continue; }
    if (use_union) unionLists(acc, cur, &tmp);
    else intersectLists(acc, cur, &tmp);
    acc.swap(tmp);
  }
  if (flipped && g_flip_convention == FlipConvention::kComplementOfBelow) {
    complementList(acc, t0, t1, out);
  } else {
    *out = acc;
  }
}

struct PlanePatch {
  double area = 0.0;
  Vec3 normal_sum{};
  Vec3 normal() const { return normalized(normal_sum); }
};

struct CellComparison {
  double symdiff = 0.0;      // |truth XOR recon| / cell volume
  double recon_vf = 0.0;     // recon volume fraction from the same columns
  std::vector<PlanePatch> planes;
};

// Integrates truth and reconstruction together over the centre cell.
// Volumes use the region's preferred axis; plane patches use all three axes
// with |n_axis| weights (see surfacePatches in geometry.h).
inline CellComparison compareCell(const Region& reg, const IRL::PlanarSeparator& sep,
                                  const Vec3& lo, const Vec3& hi, int nq) {
  CellComparison out;
  out.planes.resize(sep.getNumberOfPlanes());
  const double vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  IntervalList a, b, ab;
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    const double hu = (hi[u] - lo[u]) / nq, hv = (hi[v] - lo[v]) / nq;
    const bool volume_axis = (axis == chooseAxis(reg, 0.5 * (lo + hi)));
    for (int i = 0; i < nq; ++i) {
      for (int j = 0; j < nq; ++j) {
        Vec3 base{};
        base[u] = lo[u] + (i + 0.5) * hu;
        base[v] = lo[v] + (j + 0.5) * hv;
        separatorColumn(sep, base, axis, lo[axis], hi[axis], &b);
        if (volume_axis) {
          reg.column(base, axis, lo[axis], hi[axis], &a);
          intersectLists(a, b, &ab);
          double la = 0.0, lb = 0.0, lab = 0.0;
          for (const auto& iv : a) la += iv.hi - iv.lo;
          for (const auto& iv : b) lb += iv.hi - iv.lo;
          for (const auto& iv : ab) lab += iv.hi - iv.lo;
          out.symdiff += (la + lb - 2.0 * lab) * hu * hv;
          out.recon_vf += lb * hu * hv;
        }
        for (const auto& iv : b) {
          for (int end = 0; end < 2; ++end) {
            const int tag = end == 0 ? iv.tag_lo : iv.tag_hi;
            if (tag < 0) continue;
            const IRL::Normal& pn = sep[tag].normal();
            Vec3 n = normalized(Vec3{pn[0], pn[1], pn[2]});
            const double want = end == 0 ? -1.0 : 1.0;
            if (n[axis] * want < 0.0) n = -1.0 * n;
            const double w = std::abs(n[axis]) * hu * hv;
            out.planes[tag].area += w;
            out.planes[tag].normal_sum = out.planes[tag].normal_sum + w * n;
          }
        }
      }
    }
  }
  out.symdiff /= vol;
  out.recon_vf /= vol;
  return out;
}

inline double angleDeg(const Vec3& a, const Vec3& b) {
  const double c = std::max(-1.0, std::min(1.0, dot(normalized(a), normalized(b))));
  return std::acos(c) * 180.0 / M_PI;
}

// Picks the flip convention that reproduces IRL's own volume fraction for
// random flipped two-plane separators. Returns the mean absolute VF mismatch
// of the chosen convention (should be ~1e-4 or better).
inline double calibrateFlipConvention(unsigned long long seed) {
  std::mt19937_64 eng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
      IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
  const Vec3 lo{-0.5, -0.5, -0.5}, hi{0.5, 0.5, 0.5};
  Region dummy;
  dummy.root = dummy.prim(Quadric::plane({0.0, 0.0, 1.0}, 10.0));
  double err[2] = {0.0, 0.0};
  const int n = 50;
  for (int s = 0; s < n; ++s) {
    IRL::Normal n1(U(eng), U(eng), U(eng)), n2(U(eng), U(eng), U(eng));
    n1.normalize();
    n2.normalize();
    const IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(n1, 0.3 * U(eng)), IRL::Plane(n2, 0.3 * U(eng)), -1.0);
    const double vf_irl =
        IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep).volume() / cell.calculateVolume();
    for (int c = 0; c < 2; ++c) {
      g_flip_convention = c == 0 ? FlipConvention::kUnionBelow : FlipConvention::kComplementOfBelow;
      err[c] += std::abs(compareCell(dummy, sep, lo, hi, 64).recon_vf - vf_irl);
    }
  }
  g_flip_convention = err[0] <= err[1] ? FlipConvention::kUnionBelow : FlipConvention::kComplementOfBelow;
  return std::min(err[0], err[1]) / n;
}

}  // namespace bench

#endif  // EXAMPLES_R2PNET_BENCH_METRICS_H_
