#ifndef PLIC_GEN5X5_H_
#define PLIC_GEN5X5_H_

#include <torch/torch.h>

#include <array>
#include <functional>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/machine_learning_reconstruction/data_gen.h"
#include "irl/machine_learning_reconstruction/moments_gen.h"
#include "irl/machine_learning_reconstruction/plic_refine.h"

// Builds a fully reconstructed 3x3x3 neighborhood from a 5x5x5 moments block.
//
// The outer ring exists so that every one of the 27 inner cells has a
// complete 3x3x3 window of its own to feed the network. Each window is
// canonicalized INDEPENDENTLY, reproducing data_gen::generate()'s per-sample
// pipeline exactly (own flip decision, own reflectMoments), because that is
// the distribution the network was trained on. The predicted normal is then
// mapped back out of that window's canonical frame so all 27 planes live in
// one shared frame -- which is what the gate and fit require.
namespace plicgen5x5 {

using Predict = std::function<IRL::Normal(const torch::Tensor&)>;

constexpr int kStride = 7;

inline int flatIndex5(int i, int j, int k) { return (i * 5 + j) * 5 + k; }
inline int flatIndex3(int i, int j, int k) { return (i * 3 + j) * 3 + k; }

// --- Inverting data_gen's canonicalization on a normal --------------------
//
// Forward (data_gen::generate): canonical = flipsign(R_dir2(R_dir1(raw))),
// where flipsign negates iff !flip. Switch bodies copied verbatim from
// data_gen.h so they cannot drift from what the network was trained on.
//
// R_dir1 is always independent sign flips  -> self-inverse.
// R_dir2 cases 1-3 are single transpositions -> self-inverse.
// R_dir2 cases 4 and 5 are the two 3-cycles  -> INVERSES OF EACH OTHER.
//   Reapplying the same case does NOT undo it; see inverseDir2.
inline void applyDir1(IRL::Normal& n, int dir) {
  switch (dir) {
    case 1: n[0] = -n[0]; break;
    case 2: n[1] = -n[1]; break;
    case 3: n[2] = -n[2]; break;
    case 4: n[0] = -n[0]; n[1] = -n[1]; break;
    case 5: n[0] = -n[0]; n[2] = -n[2]; break;
    case 6: n[1] = -n[1]; n[2] = -n[2]; break;
    case 7: n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; break;
    default: break;
  }
}

inline void applyDir2(IRL::Normal& n, int dir2) {
  double t;
  switch (dir2) {
    case 1: t = n[0]; n[0] = n[1]; n[1] = t; break;
    case 2: t = n[1]; n[1] = n[2]; n[2] = t; break;
    case 3: t = n[0]; n[0] = n[2]; n[2] = t; break;
    case 4: t = n[0]; n[0] = n[1]; n[1] = t;
            t = n[1]; n[1] = n[2]; n[2] = t; break;
    case 5: t = n[0]; n[0] = n[1]; n[1] = t;
            t = n[0]; n[0] = n[2]; n[2] = t; break;
    default: break;
  }
}

inline int inverseDir2(int dir2) {
  if (dir2 == 4) return 5;
  if (dir2 == 5) return 4;
  return dir2;
}

inline IRL::Normal decanonicalize(IRL::Normal c, bool flip, int dir1,
                                  int dir2) {
  if (!flip) { c[0] = -c[0]; c[1] = -c[1]; c[2] = -c[2]; }
  applyDir2(c, inverseDir2(dir2));
  applyDir1(c, dir1);
  c.normalize();
  return c;
}

// --- Generation ----------------------------------------------------------

struct GenRange {
  double rota_l = 0.0, rota_h = 2.0 * M_PI;
  double rotb_l = -M_PI / 2.0, rotb_h = M_PI / 2.0;
  double rotc_l = 0.0, rotc_h = 2.0 * M_PI;
  double coa = 0.0, cob = 0.0;   // curvature ladder rung
  double ox_l = -0.5, ox_h = 0.5;
  double oy_l = -0.5, oy_h = 0.5;
  double oz_l = -0.5, oz_h = 0.5;
};

// Raw, unswapped septuples for the whole 5x5x5 block: sym=false so no global
// phase swap is baked in -- every window decides its own flip below.
inline std::vector<double> rawBlock(IRL::moments_gen& gen5,
                                    const IRL::Paraboloid& p) {
  bool flip_unused;
  return gen5.get_moments(p, /*order=*/1, /*sym=*/false, flip_unused);
}

// --- Per-window processing -----------------------------------------------

struct WindowResult {
  bool mixed = false;
  double true_vf = 0.0;
  std::array<std::array<double, 7>, 27> true_septuples{};
  IRL::Normal normal_raw;
};

inline WindowResult processWindow(const std::vector<double>& raw125,
                                  IRL::data_gen& dg3, const Predict& predict,
                                  int oi, int oj, int ok) {
  WindowResult out;
  const int cbase = kStride * flatIndex5(2 + oi, 2 + oj, 2 + ok);
  out.true_vf = raw125[cbase];
  out.mixed = out.true_vf > IRL::global_constants::VF_LOW &&
              out.true_vf < IRL::global_constants::VF_HIGH;
  if (!out.mixed) return out;

  std::vector<double> canon(27 * kStride);
  for (int di = 0; di < 3; ++di)
    for (int dj = 0; dj < 3; ++dj)
      for (int dk = 0; dk < 3; ++dk) {
        const int local = flatIndex3(di, dj, dk);
        const int base =
            kStride * flatIndex5(1 + oi + di, 1 + oj + dj, 1 + ok + dk);
        for (int q = 0; q < kStride; ++q) {
          out.true_septuples[local][q] = raw125[base + q];
          canon[kStride * local + q] = raw125[base + q];
        }
      }

  // This window's own flip decision, matching get_moments's rule applied to
  // THIS window's center cell -- not the 5x5x5 block's center.
  const bool window_flip = out.true_vf > 0.5;
  if (window_flip) {
    for (int c = 0; c < 27; ++c) {
      double* s = &canon[kStride * c];
      s[0] = 1.0 - s[0];
      std::swap(s[1], s[4]);
      std::swap(s[2], s[5]);
      std::swap(s[3], s[6]);
    }
  }

  // Canonicalize through data_gen's own public helpers, in generate()'s
  // order: centroid computed AFTER the flip swap, then reflectMoments.
  const IRL::Pt center = dg3.get_global_centroid(canon);
  int dir1 = 0, dir2 = 0;
  dg3.reflectMoments(canon, dir1, dir2, center);

  torch::Tensor t = torch::zeros({27 * kStride}, torch::kDouble);
  auto acc = t.accessor<double, 1>();
  for (int q = 0; q < 27 * kStride; ++q) acc[q] = canon[q];

  out.normal_raw = decanonicalize(predict(t), window_flip, dir1, dir2);
  return out;
}

// --- Neighborhood assembly ------------------------------------------------

// Center first, then the other 26 -- matching plicfit's index-0 convention.
inline const int (*neighborOffsets())[3] {
  static const int off[27][3] = {
      {0, 0, 0},
      {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0},
      {-1, 0, 1},   {-1, 1, -1}, {-1, 1, 0},  {-1, 1, 1},  {0, -1, -1},
      {0, -1, 0},   {0, -1, 1},  {0, 0, -1},  {0, 0, 1},   {0, 1, -1},
      {0, 1, 0},    {0, 1, 1},   {1, -1, -1}, {1, -1, 0},  {1, -1, 1},
      {1, 0, -1},   {1, 0, 0},   {1, 0, 1},   {1, 1, -1},  {1, 1, 0},
      {1, 1, 1}};
  return off;
}

inline void buildNeighborhood(const std::vector<double>& raw125,
                              IRL::data_gen& dg3, const Predict& predict,
                              const plicfit::Options& opt,
                              std::vector<plicfit::NeighborPlane>* planes,
                              std::vector<plicfit::StencilCell>* fit_stencil) {
  const int (*off)[3] = neighborOffsets();
  planes->assign(27, plicfit::NeighborPlane{});
  fit_stencil->clear();

  for (int m = 0; m < 27; ++m) {
    const int oi = off[m][0], oj = off[m][1], ok = off[m][2];
    plicfit::NeighborPlane& np = (*planes)[m];
    np.lo = IRL::Pt(oi - 0.5, oj - 0.5, ok - 0.5);
    np.hi = IRL::Pt(oi + 0.5, oj + 0.5, ok + 0.5);

    const WindowResult w = processWindow(raw125, dg3, predict, oi, oj, ok);
    np.mixed = w.mixed;
    if (!w.mixed) continue;
    np.normal = w.normal_raw;

    IRL::RectangularCuboid cell =
        IRL::RectangularCuboid::fromBoundingPts(np.lo, np.hi);
    IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromOnePlane(
        IRL::Plane(np.normal, np.normal * IRL::Pt(oi, oj, ok)));
    IRL::setDistanceToMatchVolumeFraction(cell, w.true_vf, &sep,
                                          opt.volume_tolerance);
    np.distance = sep[0].distance();

    if (m == 0) {
      // Center: assemble the 7-point face stencil for the LM fit, from the
      // TRUE unswapped septuples. The fit needs physical moments, never the
      // network's canonicalized view of them.
      static const int face_local[7][3] = {{1, 1, 1}, {0, 1, 1}, {2, 1, 1},
                                           {1, 0, 1}, {1, 2, 1}, {1, 1, 0},
                                           {1, 1, 2}};
      fit_stencil->assign(7, plicfit::StencilCell{});
      for (int f = 0; f < 7; ++f) {
        const int li = face_local[f][0], lj = face_local[f][1],
                  lk = face_local[f][2];
        const auto& s = w.true_septuples[flatIndex3(li, lj, lk)];
        plicfit::StencilCell& sc = (*fit_stencil)[f];
        sc.cell = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt(li - 1.5, lj - 1.5, lk - 1.5),
            IRL::Pt(li - 0.5, lj - 0.5, lk - 0.5));
        sc.volume = 1.0;
        sc.vf = s[0];
        const IRL::Pt cc = sc.cell.calculateCentroid();
        sc.liquid_bary = IRL::Pt(cc[0] + s[1], cc[1] + s[2], cc[2] + s[3]);
        sc.gas_bary = IRL::Pt(cc[0] + s[4], cc[1] + s[5], cc[2] + s[6]);
        // get_moments writes six literal zeros for a pure cell, which in this
        // relative frame reads as "cell center" -- a sentinel, not a centroid.
        sc.liquid_meaningful = s[0] > IRL::global_constants::VF_LOW &&
                               s[0] < IRL::global_constants::VF_HIGH;
        sc.gas_meaningful = sc.liquid_meaningful;
      }
    }
  }
}

// --- Ground truth ---------------------------------------------------------

// Exact surface-averaged normal of the paraboloid over the real cell -- no
// dependence on reflectMoments or flip, computed fresh from geometry. Sign
// convention differs from data_gen's, so compare over both signs.
inline IRL::Normal exactSurfaceNormal(IRL::moments_gen& gen5,
                                      const IRL::Paraboloid& p, int i, int j,
                                      int k) {
  auto cell = gen5.getStencil()->getCell(i, j, k);
  auto sm = IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments,
                            IRL::ParaboloidParametrizedSurfaceOutput>>(cell, p);
  IRL::Normal n = sm.getSurface().getAverageNormalNonAligned();
  n.normalize();
  return n;
}

// --- Per-sample run -------------------------------------------------------

struct SampleResult {
  bool valid = false;        // false => center cell pure, sample skipped
  bool gated_in = false;
  double flatness = 0.0;
  double residual_before = 0.0;
  double residual_after = 0.0;
  double angle_raw_deg = 0.0;
  double angle_fit_deg = 0.0;
  IRL::Normal normal_truth;   // exact surface normal (sign per exactSurfaceNormal)
  IRL::Normal normal_raw;     // network's own prediction, shared frame
  IRL::Normal normal_fit;     // after gate+fit (== normal_raw if gated out)
};

inline SampleResult runSample(IRL::moments_gen& gen5, IRL::data_gen& dg3,
                              const GenRange& r, const Predict& predict,
                              const plicfit::Options& opt) {
  SampleResult out;

  const IRL::Paraboloid p = gen5.new_random_paraboloid(
      r.rota_l, r.rota_h, r.rotb_l, r.rotb_h, r.rotc_l, r.rotc_h, r.coa, r.coa,
      r.cob, r.cob, r.ox_l, r.ox_h, r.oy_l, r.oy_h, r.oz_l, r.oz_h);
  const std::vector<double> raw125 = rawBlock(gen5, p);

  std::vector<plicfit::NeighborPlane> planes;
  std::vector<plicfit::StencilCell> fit_stencil;
  buildNeighborhood(raw125, dg3, predict, opt, &planes, &fit_stencil);

  if (!planes[0].mixed || fit_stencil.empty()) return out;
  out.valid = true;
  out.normal_raw = planes[0].normal;

  {
    plicfit::PlaneFit probe(fit_stencil, planes[0].normal, opt);
    std::array<double, 2> zero = {0.0, 0.0};
    const auto r0 = probe.residual(zero);
    double s2 = 0.0;
    for (int i = 0; i < probe.numFitRows(); ++i) s2 += r0[i] * r0[i];
    out.residual_before = std::sqrt(s2);
  }

  IRL::Normal fitted = planes[0].normal;
  const plicfit::Result res =
      (opt.orientation_method == 0)
          ? plicfit::refine(fit_stencil, planes, fitted, opt)
          : plicfit::refineParaboloid(planes, fitted, opt);
  out.gated_in = res.fitted;
  out.flatness = res.flatness;
  out.normal_fit = fitted;

  // Recompute the fit-row residual through a PlaneFit probe on the
  // paraboloid path, so the resid_before/resid_after columns mean the same
  // thing on both branches (refineParaboloid's own residual is a weighted LS
  // height residual, not comparable).
  if (opt.orientation_method != 0 && res.fitted) {
    plicfit::PlaneFit probe(fit_stencil, fitted, opt);
    std::array<double, 2> zero = {0.0, 0.0};
    const auto rr = probe.residual(zero);
    double s2 = 0.0;
    for (int i = 0; i < probe.numFitRows(); ++i) s2 += rr[i] * rr[i];
    out.residual_after = std::sqrt(s2);
  } else {
    out.residual_after = res.fitted ? res.residual : out.residual_before;
  }

  const IRL::Normal truth = exactSurfaceNormal(gen5, p, 2, 2, 2);
  out.normal_truth = truth;
  auto angleDeg = [&truth](const IRL::Normal& n) {
    const double c = std::clamp(n * truth, -1.0, 1.0);
    return std::min(std::acos(c), std::acos(-c)) * 180.0 / M_PI;
  };
  out.angle_raw_deg = angleDeg(planes[0].normal);
  out.angle_fit_deg = angleDeg(fitted);
  return out;
}

}  // namespace plicgen5x5

#endif  // PLIC_GEN5X5_H_