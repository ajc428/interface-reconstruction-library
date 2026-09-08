#ifndef R2P_COUPLED_FIT_H_
#define R2P_COUPLED_FIT_H_

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"

#include "irl/machine_learning_reconstruction/plic_paraboloid.h"

// Coupled two-surface paraboloid fit.
//
// PROBLEM. Fitting each surface of a sheet independently gives each its own
// frame and its own free tilt, so the WEDGE ANGLE between the two planes is
// unconstrained -- it can change by up to 2*max_rotation with nothing
// watching it. In a flat sheet interior the two faces are near-antiparallel
// (a slab, wedge angle ~0) and the quadratic terms have no real signal to
// fit, so a1/a2 pick up noise. A small spurious tilt on one face opens a
// wedge that was not there; R2PDistanceSolver then re-derives the plane pair
// from the new bisector and a plane that previously missed the cell starts
// clipping a corner. Volume is conserved exactly throughout, so nothing
// downstream flags it.
//
// MODEL. One frame, built from the center polygon of group 0. Both surfaces
// are height fields over the same (t,s) tangent plane:
//
//   group 0:  n = a0_0 + (a1+d1_0) t + (a2+d2_0) s + a3 t^2 + a4 ts + a5 s^2
//   group 1:  n = a0_1 + (a1+d1_1) t + (a2+d2_1) s + a3 t^2 + a4 ts + a5 s^2
//
// Shared a1..a5: common shape. Separate a0_g: the two faces sit at different
// heights, and a0_1 - a0_0 IS the sheet thickness. The per-group linear
// corrections d1_g, d2_g are the splay, penalized by `splay_penalty`. At
// splay_penalty -> infinity the faces are exactly parallel, which is the
// correct answer for a sheet interior and the configuration that cannot
// produce a spurious plane. At splay_penalty -> 0 this reduces to two
// independent fits sharing only curvature.
namespace r2pcouple {

struct Options {
  int method = 1;                 // 1 = pointwise rows, 2 = integral rows
  double h = 2.5;                 // wgauss support radius, in mesh_size units
  double mesh_size = 1.0;         // pass the cell width
  double splay_penalty = 0.015;
  // Ridge weight on d1_g, d2_g relative to total row weight. 25 is stiff --
  // splay must be strongly supported by the data to survive. Lower toward 1
  // if sheet ends come out under-splayed; raise if spurious planes persist.

  double curvature_penalty = 1.0e-3;
  // Mild ridge on a3, a4, a5. A nearly-collinear sample set otherwise
  // produces wild curvature that leaks into the linear terms through the
  // shared columns.

  int min_per_group = 6;
  double max_resid = 0.25;

  // FRAME SOURCE. Which direction and origin define the shared tangent
  // plane. kGroup0 -- the original behaviour and still the default -- uses
  // group 0's center polygon, which makes the estimator depend on an
  // arbitrary labeling: swap the two groups and the answer changes. The
  // bisector options are label-symmetric.
  //
  // They also halve the maximum slope either surface has to express: with a
  // group-0 frame the other surface sits at tan(splay), with a bisector
  // frame both sit at tan(splay/2). At 90 degrees of splay the group-0 frame
  // puts group 1's normal exactly perpendicular to the frame, which trips the
  // |nloc[2]| < 1e-10 guard in the integral branch and discards the polygon;
  // past 90 degrees the facing test rejects group 1 wholesale. The bisector
  // has no such cliff. This matters most at sheet ends.
  //
  //   kGroup0        - original behaviour (default: no change on rebuild)
  //   kBisector      - unweighted: nref ~ n0 - n1
  //   kAreaWeighted  - center-polygon areas weight the two contributions
  //
  // Of the two new modes kAreaWeighted is the safer: when one group's center
  // polygon is a sliver its normal is the least trustworthy input in the fit,
  // and an unweighted average gives it equal say.
  enum class FrameSource { kGroup0, kBisector, kAreaWeighted };
  FrameSource frame_source = FrameSource::kBisector;

  bool bisector_origin = true;
  // Only consulted when frame_source != kGroup0. Moves pref to the matching
  // weighted midpoint of the two center centroids, so a0_0 and a0_1 come out
  // roughly antisymmetric about zero rather than one being ~0 and the other
  // carrying the whole thickness. thickness = |a0_1 - a0_0| is unchanged
  // either way. Set false to rotate the frame but keep the origin on group
  // 0's centroid, which isolates the direction change if you want to A/B the
  // two effects separately.

  double min_bisector_magnitude = 1.0e-3;
  // |w0*n0 - w1*n1| below this means the two normals are nearly PARALLEL
  // rather than antiparallel -- the groups have collapsed onto the same
  // surface, or the sort failed. The bisector is meaningless there, so fall
  // back to the group-0 frame rather than normalizing noise.

  bool planarity_snap = false;
  double flat_curvature_tol = 0.00;
  // PLANARITY SNAP. If the fitted mean curvature of BOTH surfaces is below
  // flat_curvature_tol (dimensionless: curvature * mesh_size), the surface is
  // flat to within what this mesh can resolve, and the quadratic terms are
  // fitting nothing but the scatter in the input PLIC centroids. Refitting
  // with the curvature block pinned to zero then recovers the exact plane,
  // because centroids of planar polygons lie exactly on their own plane -- so
  // a pure linear fit through them is exact once the inputs are consistent.
  //
  // This is what restores flat-sheet behaviour. Without it, a flat diagonal
  // sheet keeps absorbing input error into a3/a4/a5 and the extracted normal
  // stays slightly off forever.
};

struct Result {
  IRL::Normal normal[2];
  double curvature[2] = {0.0, 0.0};
  double offset[2] = {0.0, 0.0};     // a0_g, in mesh_size units
  double thickness = 0.0;            // |a0_1 - a0_0| * mesh_size
  double splay_angle = 0.0;          // radians between the two fitted normals
  double rms_residual = 0.0;
  int count[2] = {0, 0};
  bool snapped_to_planar = false;
  IRL::Normal nref, tref, sref;
};

namespace detail {

// Number of unknowns: [a0_0, a0_1, a1, a2, a3, a4, a5, d1_0, d2_0, d1_1, d2_1]
constexpr int kNumUnknowns = 11;
// 4 splay penalty rows + 3 curvature penalty rows.
constexpr int kNumPenaltyRows = 7;

using CoeffVector = Eigen::Matrix<double, kNumUnknowns, 1>;
using MonomialVector = Eigen::Matrix<double, 6, 1>;

// The 6 monomial integrals [1, t, s, t^2, ts, s^2] over a polygon's projected
// footprint, by Green's theorem on the boundary. Transcribed from
// plicparab::fitIntegral so the coupled system can assemble the same rows.
// If you change one, change both -- or factor this out of plic_paraboloid.h
// and have both call it.
inline MonomialVector polygonIntegrals(const plicparab::SurfacePolygon& sp,
                                       const IRL::Pt& pref,
                                       const IRL::Normal& tref,
                                       const IRL::Normal& sref,
                                       const double mesh_size) {
  MonomialVector integrals = MonomialVector::Zero();
  const int shape = static_cast<int>(sp.vertices.size());
  for (int v = 0; v < shape; ++v) {
    const int vn = (v + 1) % shape;
    const IRL::Pt d1((sp.vertices[v][0] - pref[0]) / mesh_size,
                     (sp.vertices[v][1] - pref[1]) / mesh_size,
                     (sp.vertices[v][2] - pref[2]) / mesh_size);
    const IRL::Pt d2((sp.vertices[vn][0] - pref[0]) / mesh_size,
                     (sp.vertices[vn][1] - pref[1]) / mesh_size,
                     (sp.vertices[vn][2] - pref[2]) / mesh_size);
    const double xv = tref * d1, yv = sref * d1;
    const double xvn = tref * d2, yvn = sref * d2;

    integrals(0) += (xv * yvn - xvn * yv) / 2.0;
    integrals(1) += (xv + xvn) * (xv * yvn - xvn * yv) / 6.0;
    integrals(2) += (yv + yvn) * (xv * yvn - xvn * yv) / 6.0;
    integrals(3) += (xv + xvn) * (xv * xv + xvn * xvn) * (yvn - yv) / 12.0;
    integrals(4) += (yvn - yv) *
                    (3.0 * xv * xv * yv + xv * xv * yvn + 2.0 * xv * xvn * yv +
                     2.0 * xv * xvn * yvn + xvn * xvn * yv +
                     3.0 * xvn * xvn * yvn) / 24.0;
    integrals(5) += (xv - xvn) * (yv + yvn) * (yv * yv + yvn * yvn) / 12.0;
  }
  return integrals;
}

}  // namespace detail

// polys0/polys1 must each have that group's CENTER-CELL polygon at index 0.
// seed0/seed1 are the R2PNet normals, used only to resolve sign.
inline std::optional<Result> fitCoupled(
    const std::vector<plicparab::SurfacePolygon>& polys0,
    const std::vector<plicparab::SurfacePolygon>& polys1,
    const IRL::Normal& seed0, const IRL::Normal& seed1,
    const Options& opt = Options()) {
  if (static_cast<int>(polys0.size()) < opt.min_per_group ||
      static_cast<int>(polys1.size()) < opt.min_per_group) {
    return std::nullopt;
  }

  // ONE frame for both surfaces. Group 1's polygons face roughly -nref, but
  // they are still a single-valued height field over the same (t,s) plane,
  // which is all the fit requires. The sign is restored at extraction.
  //
  // Group 0's normal points along +nref and group 1's along -nref, so the
  // direction that splits them symmetrically is (n0 - n1), not their sum.
  IRL::Pt pref = polys0[0].centroid;
  IRL::Normal seed_normal = polys0[0].normal;

  if (opt.frame_source != Options::FrameSource::kGroup0) {
    IRL::Normal n0 = polys0[0].normal;
    IRL::Normal n1 = polys1[0].normal;
    n0.normalize();
    n1.normalize();

    double w0 = 0.5;
    double w1 = 0.5;
    if (opt.frame_source == Options::FrameSource::kAreaWeighted) {
      const double a0_area = polys0[0].area;
      const double a1_area = polys1[0].area;
      const double wsum = a0_area + a1_area;
      if (wsum > 0.0) {
        w0 = a0_area / wsum;
        w1 = a1_area / wsum;
      }
    }

    const IRL::Normal combined = w0 * n0 - w1 * n1;
    if (combined.calculateMagnitude() >= opt.min_bisector_magnitude) {
      seed_normal = combined;
      if (opt.bisector_origin) {
        pref = IRL::Pt(w0 * polys0[0].centroid[0] + w1 * polys1[0].centroid[0],
                       w0 * polys0[0].centroid[1] + w1 * polys1[0].centroid[1],
                       w0 * polys0[0].centroid[2] + w1 * polys1[0].centroid[2]);
      }
    }
    // else: groups are near-parallel, so they have collapsed onto one surface
    // or the sort failed. Keep the group-0 frame; the area-fraction gate
    // upstream and max_resid below are the checks that catch that case.
  }

  IRL::Normal nref, tref, sref;
  plicparab::buildFrame(seed_normal, &nref, &tref, &sref);

  std::vector<detail::CoeffVector> rows;
  std::vector<double> rhs;
  double weight_total = 0.0;
  int count[2] = {0, 0};

  for (int g = 0; g < 2; ++g) {
    const std::vector<plicparab::SurfacePolygon>& polys = (g == 0) ? polys0 : polys1;
    // Group 1's surface legitimately faces away from nref, so every
    // orientation test for that group is taken against -nref. Taking it
    // against nref (as plicparab::fitIntegral does, since it only ever sees
    // one surface) would discard every group-1 polygon.
    const IRL::Normal facing = (g == 0) ? nref : -nref;

    for (std::size_t idx = 0; idx < polys.size(); ++idx) {
      const plicparab::SurfacePolygon& sp = polys[idx];
      IRL::Normal nglob = sp.normal;
      nglob.normalize();

      const IRL::Pt dc((sp.centroid[0] - pref[0]) / opt.mesh_size,
                       (sp.centroid[1] - pref[1]) / opt.mesh_size,
                       (sp.centroid[2] - pref[2]) / opt.mesh_size);
      const double pt = tref * dc;
      const double ps = sref * dc;
      const double pn = nref * dc;
      const double dist = std::sqrt(pt * pt + ps * ps + pn * pn);
      const double wg = plicparab::wgauss(dist, opt.h);
      if (wg <= 0.0) continue;

      detail::MonomialVector m = detail::MonomialVector::Zero();
      double r = 0.0;
      double w = 0.0;

      if (opt.method == 2) {
        if (nglob * facing <= 0.0) continue;

        const IRL::Normal nloc(tref * nglob, sref * nglob, nref * nglob);
        if (std::abs(nloc[2]) < 1.0e-10) continue;

        const double ndotp = nloc[0] * pt + nloc[1] * ps + nloc[2] * pn;
        const double inv = 1.0 / (-nloc[2]);
        const double c0 = -ndotp * inv;
        const double c1 = nloc[0] * inv;
        const double c2 = nloc[1] * inv;

        m = detail::polygonIntegrals(sp, pref, tref, sref, opt.mesh_size);
        r = c0 * m(0) + c1 * m(1) + c2 * m(2);
        w = wg;
      } else {
        const double align = std::max(nglob * facing, 0.0);
        const double surf = sp.area / (opt.mesh_size * opt.mesh_size);
        w = surf * align * wg;
        if (w <= 0.0) continue;
        m(0) = 1.0;
        m(1) = pt;
        m(2) = ps;
        m(3) = pt * pt;
        m(4) = pt * ps;
        m(5) = ps * ps;
        r = pn;
      }

      const double sw = std::sqrt(w);
      detail::CoeffVector row = detail::CoeffVector::Zero();
      row(g) = sw * m(0);                  // a0_0 or a0_1
      row(2) = sw * m(1);                  // a1  (shared)
      row(3) = sw * m(2);                  // a2  (shared)
      row(4) = sw * m(3);                  // a3  (shared)
      row(5) = sw * m(4);                  // a4  (shared)
      row(6) = sw * m(5);                  // a5  (shared)
      row(7 + 2 * g) = sw * m(1);          // d1_g (splay)
      row(8 + 2 * g) = sw * m(2);          // d2_g (splay)

      rows.push_back(row);
      rhs.push_back(sw * r);
      weight_total += w;
      ++count[g];
    }
  }

  if (count[0] < opt.min_per_group || count[1] < opt.min_per_group) {
    return std::nullopt;
  }

  const int ndata = static_cast<int>(rows.size());
  const int ntotal = ndata + detail::kNumPenaltyRows;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ntotal, detail::kNumUnknowns);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(ntotal);
  for (int i = 0; i < ndata; ++i) {
    A.row(i) = rows[i].transpose();
    b(i) = rhs[i];
  }

  // Penalty rows, scaled by total row weight so they hold their relative
  // strength regardless of stencil size. RHS stays zero: these pull the
  // penalized coefficients toward zero.
  const double splay_w = std::sqrt(opt.splay_penalty * weight_total);
  const double curv_w = std::sqrt(opt.curvature_penalty * weight_total);
  int prow = ndata;
  for (int c = 7; c < detail::kNumUnknowns; ++c) A(prow++, c) = splay_w;
  for (int c = 4; c < 7; ++c) A(prow++, c) = curv_w;

  const Eigen::VectorXd sol_dyn =
      A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  if (!sol_dyn.allFinite()) return std::nullopt;

  // Fixed-size copy so the residual dot products below are compile-time sized.
  detail::CoeffVector sol;
  for (int c = 0; c < detail::kNumUnknowns; ++c) sol(c) = sol_dyn(c);

  Result out;
  out.count[0] = count[0];
  out.count[1] = count[1];
  out.nref = nref;
  out.tref = tref;
  out.sref = sref;

  double s2 = 0.0;
  for (int i = 0; i < ndata; ++i) {
    const double res = rows[i].dot(sol) - rhs[i];
    s2 += res * res;
  }
  out.rms_residual = std::sqrt(s2 / static_cast<double>(ndata));
  if (out.rms_residual > opt.max_resid) return std::nullopt;

  const double a3 = sol(4);
  const double a4 = sol(5);
  const double a5 = sol(6);
  for (int g = 0; g < 2; ++g) {
    const double ft = sol(2) + sol(7 + 2 * g);
    const double fs = sol(3) + sol(8 + 2 * g);

    IRL::Normal fitted = nref - ft * tref - fs * sref;
    fitted.normalize();
    // Group 1's surface faces the other way; restore sign against the seed.
    const IRL::Normal& seed = (g == 0) ? seed0 : seed1;
    if (fitted * seed < 0.0) fitted = -fitted;
    out.normal[g] = fitted;

    // Same curvature expression as plicparab::finishFit, with the [1,t,s,
    // t^2,ts,s^2] basis putting the 2x on the pure second derivatives.
    const double denom = std::pow(1.0 + ft * ft + fs * fs, 1.5);
    out.curvature[g] = -((1.0 + ft * ft) * (2.0 * a5) -
                         2.0 * ft * fs * a4 +
                         (1.0 + fs * fs) * (2.0 * a3)) / denom / opt.mesh_size;
    out.offset[g] = sol(g);
  }

  out.thickness = std::abs(sol(1) - sol(0)) * opt.mesh_size;
  const double opposed = -(out.normal[0] * out.normal[1]);
  out.splay_angle = std::acos(std::max(-1.0, std::min(1.0, opposed)));

  // Planarity snap: both surfaces flat to mesh resolution -> refit with the
  // curvature block effectively pinned to zero. Recurses once, with the snap
  // disabled so it cannot loop.
  if (opt.planarity_snap) {
    const double c0 = std::abs(out.curvature[0]) * opt.mesh_size;
    const double c1 = std::abs(out.curvature[1]) * opt.mesh_size;
    if (c0 < opt.flat_curvature_tol && c1 < opt.flat_curvature_tol) {
      Options flat = opt;
      flat.planarity_snap = false;
      flat.curvature_penalty = 1.0e8;
      std::optional<Result> planar =
          fitCoupled(polys0, polys1, seed0, seed1, flat);
      if (planar) {
        planar->snapped_to_planar = true;
        return planar;
      }
    }
  }
  return out;
}

}  // namespace r2pcouple

#endif  // R2P_COUPLED_FIT_H_