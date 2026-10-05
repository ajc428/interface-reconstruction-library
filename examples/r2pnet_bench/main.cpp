// R2P-Net benchmark.
//
// Samples exact film geometries by category, computes the 9^3 block of
// moments the reconstruction sees, runs each method on the centre cell and
// compares against the true geometry.
//
//   r2pnet_bench [--cat all|a,b,...] [--n N] [--seed S] [--noise SIGMA]
//                [--methods m1,m2,...] [--nq NQ] [--out FILE.csv]
//   r2pnet_bench --selftest
//   r2pnet_bench --summarize FILE.csv [FILE.csv ...]
//
// Methods:
//   plicnet          PLICNet single plane (the PLIC branch of R2P3D_Net)
//   r2pnet           R2P-Net branch of R2P3D_Net on the centre cell, always
//   pipeline_nopass  classifier routing + both branches, as deployed, no pass 2
//   pipeline         the above plus the paraboloid pass (the current best)
//   oracle           exact face normals + R2PDistanceSolver (normal-error-free)
//   oracle_cm        exact face normals + VF-and-centroid distance solve
//   r2p_mof          IRL R2P optimizer seeded with MOF (non-ML reference)
// Any of r2pnet, pipeline_nopass and pipeline may be suffixed with _cm, which
// swaps R2PDistanceSolver for the volume-fraction-and-centroid solve in
// distance_solve.h -- everywhere it is called, including inside r2ppass.
// The _nw suffix does the same with the Newton solver in newton_solve.h.
// Per-sample rows go to the CSV; a per-category summary goes to stdout.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "examples/r2pnet_bench/geometry.h"
#include "examples/r2pnet_bench/methods.h"
#include "examples/r2pnet_bench/metrics.h"
#include "examples/r2pnet_bench/scenes.h"

using namespace bench;

namespace {

constexpr double kFaceArea = 1.0e-3;   // truth patch counts as a face above this (cell-face units)

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
  return out;
}

struct Row {
  std::string cat, method;
  long sample = 0;
  double thickness = 0, r1 = 0, r2 = 0, param = 0, vf = 0;
  int n_true = 0, n_pred = 0;
  double area1 = 0, area2 = 0;   // two largest true face areas
  double symdiff = 0, cent_err = 0, ang_mean = 0, ang_max = 0, vf_err = 0;
  int cls = -1, routed = -1;
  double aux1 = 0, aux2 = 0;     // method-specific (r2pnet: |n1|, |n2|)
  double ms = 0;
  int rule_flip = -1;   // deployed flip rule at the centre (1 = gas is phase 0)
  int pca_flip = -1;    // PCA rule at the centre
};

const char* kHeader =
    "category,sample,method,thickness,radius1,radius2,param,vf,n_true,n_pred,area1,area2,"
    "symdiff,centroid_err,angle_mean_deg,angle_max_deg,class,routed_r2p,aux1,aux2,vf_err,ms,rule_flip,pca_flip";

void writeRow(std::ostream& o, const Row& r) {
  o << r.cat << ',' << r.sample << ',' << r.method << ',' << r.thickness << ',' << r.r1 << ','
    << r.r2 << ',' << r.param << ',' << r.vf << ',' << r.n_true << ',' << r.n_pred << ','
    << r.area1 << ',' << r.area2 << ',' << r.symdiff << ',' << r.cent_err << ',' << r.ang_mean
    << ',' << r.ang_max << ',' << r.cls << ',' << r.routed << ',' << r.aux1 << ',' << r.aux2
    << ',' << r.vf_err << ',' << r.ms << ',' << r.rule_flip << ',' << r.pca_flip << '\n';
}

struct TruthFace {
  Vec3 normal;
  double area;
};

// Fills the metric fields of a row for one reconstruction of the centre cell.
void score(const Region& reg, const std::vector<TruthFace>& faces, const IRL::Pt& true_liq,
           const IRL::PlanarSeparator& sep, int nq, Row* row) {
  const Vec3 lo{-0.5, -0.5, -0.5}, hi{0.5, 0.5, 0.5};
  const CellComparison cmp = compareCell(reg, sep, lo, hi, nq);
  row->symdiff = cmp.symdiff;
  // Volume conservation: every method must reproduce the cell's volume
  // fraction exactly. A method that quietly empties the cell would otherwise
  // only show up as a middling symmetric difference. Measured with IRL's own
  // cut rather than the column integration, whose noise floor (~1e-4) would
  // swamp the quantity being checked.

  row->n_pred = 0;
  for (const auto& p : cmp.planes) if (p.area > kFaceArea) ++row->n_pred;

  // Area-weighted angle over true faces, each matched to the reconstructed
  // patch with the closest orientation.
  double wsum = 0.0, asum = 0.0, amax = 0.0;
  for (const auto& f : faces) {
    if (f.area <= kFaceArea) continue;
    double best = 180.0;
    for (const auto& p : cmp.planes)
      if (p.area > 1.0e-6) best = std::min(best, angleDeg(f.normal, p.normal()));
    asum += f.area * best;
    wsum += f.area;
    amax = std::max(amax, best);
  }
  row->ang_mean = wsum > 0.0 ? asum / wsum : 0.0;
  row->ang_max = amax;

  const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
      IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
  const auto m = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep);
  row->cent_err = m.volume() > 1.0e-14 ? IRL::magnitude(m.centroid() - true_liq) : 0.0;
  row->vf_err = std::abs(m.volume() / cell.calculateVolume() - row->vf);
}

// Two-face oracle: true normals, same distance solve as R2P-Net.
IRL::PlanarSeparator oracle(const Block& blk, const Region& reg, const std::vector<TruthFace>& faces,
                            int nq) {
  std::vector<TruthFace> f;
  for (const auto& x : faces) if (x.area > kFaceArea) f.push_back(x);
  if (f.empty() && !faces.empty()) f.push_back(faces[0]);
  if (f.empty()) {
    return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), 1.0));
  }
  const IRL::RectangularCuboid cell = blk.cell(Block::kC, Block::kC, Block::kC);
  const double vf = blk.vf(Block::kC, Block::kC, Block::kC);
  auto toN = [](const Vec3& v) { return IRL::Normal(v[0], v[1], v[2]); };
  if (f.size() == 1) {
    const IRL::Normal n = toN(f[0].normal);
    return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, IRL::findDistanceOnePlane(cell, vf, n)));
  }
  IRL::PlanarSeparator best;
  double best_err = 1.0e300;
  for (double flip : {1.0, -1.0}) {
    IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(toN(f[0].normal), 0.0), IRL::Plane(toN(f[1].normal), 0.0), flip);
    R2PDistanceSolverImpl(vf, blk.liq(Block::kC, Block::kC, Block::kC), sep, cell);
    const double e = compareCell(reg, sep, {-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}, nq).symdiff;
    if (e < best_err) { best_err = e; best = sep; }
  }
  if (best_err >= 1.0e300) {
    // Both candidates degenerate: fall back to the dominant face alone.
    const IRL::Normal n = toN(f[0].normal);
    return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, IRL::findDistanceOnePlane(cell, vf, n)));
  }
  return best;
}

// Oracle normals with the moment-matching distance solve above.
IRL::PlanarSeparator oracleCM(const Block& blk, const Region& reg, const std::vector<TruthFace>& faces,
                              int nq) {
  std::vector<TruthFace> f;
  for (const auto& x : faces) if (x.area > kFaceArea) f.push_back(x);
  if (f.empty() && !faces.empty()) f.push_back(faces[0]);
  const IRL::RectangularCuboid cell = blk.cell(Block::kC, Block::kC, Block::kC);
  const double vf = blk.vf(Block::kC, Block::kC, Block::kC);
  if (f.empty()) return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), 1.0));
  auto toN = [](const Vec3& v) { return IRL::Normal(v[0], v[1], v[2]); };
  if (f.size() == 1) {
    const IRL::Normal n = toN(f[0].normal);
    return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, IRL::findDistanceOnePlane(cell, vf, n)));
  }
  IRL::PlanarSeparator best;
  double best_err = 1.0e300;
  for (double flip : {1.0, -1.0}) {
    const IRL::PlanarSeparator sep = solveTwoDistances(toN(f[0].normal), toN(f[1].normal), flip, vf,
                                                       blk.liq(Block::kC, Block::kC, Block::kC), cell);
    const double e = compareCell(reg, sep, {-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}, nq).symdiff;
    if (e < best_err) { best_err = e; best = sep; }
  }
  return best;
}

// Analytic Jacobian against finite differences, for both flip states.
int jacobianCheck(long n, unsigned long long seed) {
  std::mt19937_64 eng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
      IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
  const IRL::Pt lo(-0.5, -0.5, -0.5), hi(0.5, 0.5, 0.5);
  const double vol = cell.calculateVolume();
  double worst[2] = {0.0, 0.0};
  double worst_c[2] = {0.0, 0.0};
  for (long t = 0; t < n; ++t) {
    IRL::Normal n0(U(eng), U(eng), U(eng));
    n0.normalize();
    IRL::Normal n1 = -n0;
    // Tilt the second face slightly, as a real film would be.
    n1[0] += 0.15 * U(eng); n1[1] += 0.15 * U(eng); n1[2] += 0.15 * U(eng);
    n1.normalize();
    const double d0 = 0.2 * U(eng), d1 = 0.2 * U(eng);
    for (int fi = 0; fi < 2; ++fi) {
      const double flip = fi == 0 ? 1.0 : -1.0;
      auto sepAt = [&](double e0, double e1) {
        return IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(n0, d0 + e0), IRL::Plane(n1, d1 + e1), flip);
      };
      const CutState st = evaluateCut(sepAt(0, 0), cell, lo, hi);
      if (st.volume < 0.05 * vol || st.volume > 0.95 * vol) continue;
      const double h = 1.0e-6;
      IRL::Normal m = n0 - n1;
      m.normalize();
      auto proj = [&m](const IRL::Pt& a, const IRL::Pt& b) {
        return m[0] * (a[0] - b[0]) + m[1] * (a[1] - b[1]) + m[2] * (a[2] - b[2]);
      };
      for (int p = 0; p < 2; ++p) {
        const auto mp = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
            cell, sepAt(p == 0 ? h : 0.0, p == 1 ? h : 0.0));
        const auto mm = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(
            cell, sepAt(p == 0 ? -h : 0.0, p == 1 ? -h : 0.0));
        const double fd = (mp.volume() - mm.volume()) / (2.0 * h);
        worst[fi] = std::max(worst[fi], std::abs(fd - st.area[p]) / std::max(1.0, std::abs(fd)));
        // Centroid row: d(m.C)/dd_p
        const double cp = m * mp.centroid(), cm_ = m * mm.centroid();
        const double fd_c = (cp - cm_) / (2.0 * h);
        const double an_c = st.area[p] * proj(st.poly_centroid[p], st.centroid) / st.volume;
        worst_c[fi] = std::max(worst_c[fi], std::abs(fd_c - an_c) / std::max(1.0, std::abs(fd_c)));
      }
    }
  }
  std::printf("dV/dd     vs finite difference:  unflipped %.3e   flipped %.3e\n", worst[0], worst[1]);
  std::printf("d(m.C)/dd vs finite difference:  unflipped %.3e   flipped %.3e\n", worst_c[0], worst_c[1]);
  return (worst[0] < 1e-4 && worst[1] < 1e-4 && worst_c[0] < 1e-4 && worst_c[1] < 1e-4) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Newton vs bracketed solver: same normals, same targets, same cells.
// Reports how far apart the two answers are, how well each satisfies the two
// constraints it is solving, and what each costs in cell cuts.
// ---------------------------------------------------------------------------
int validateSolver(long n, unsigned long long seed, double noise) {
  calibrateFlipConvention(7);
  SceneSampler sampler(seed);
  const Vec3 lo{-0.5, -0.5, -0.5}, hi{0.5, 0.5, 0.5};
  std::vector<double> dist_diff, res_vf_nw, res_cm_nw, res_vf_br, res_cm_br, res_vf_bad, res_cm_bad;
  long pairs = 0, newton_iters = 0, newton_failed = 0;
  long fail_by_flip[2] = {0, 0}, tot_by_flip[2] = {0, 0};
  std::map<std::string, long> fail_by_cat, tot_by_cat;

  for (const auto& cat : allCategories()) {
    for (long s = 0; s < n; ++s) {
      Scene scene = sampler.sample(cat);
      Block blk;
      fillBlock(scene.region, 32, &blk);
      if (!blk.mixed(Block::kC, Block::kC, Block::kC)) { --s; continue; }
      perturbBlock(noise, sampler.engine(), &blk);

      std::vector<TruthFace> faces;
      for (const auto& p : surfacePatches(scene.region, lo, hi, 64))
        if (p.area > kFaceArea) faces.push_back({p.normal(), p.area});
      if (faces.size() < 2) continue;
      std::sort(faces.begin(), faces.end(),
                [](const TruthFace& x, const TruthFace& y) { return x.area > y.area; });

      const IRL::RectangularCuboid cell = blk.cell(Block::kC, Block::kC, Block::kC);
      const double vf = blk.vf(Block::kC, Block::kC, Block::kC);
      const IRL::Pt liq = blk.liq(Block::kC, Block::kC, Block::kC);
      const IRL::Normal n0(faces[0].normal[0], faces[0].normal[1], faces[0].normal[2]);
      const IRL::Normal n1(faces[1].normal[0], faces[1].normal[1], faces[1].normal[2]);
      IRL::Normal m = n0 - n1;
      if (m.calculateMagnitude() < 1.0e-12) m = n0;
      m.normalize();
      const double target = m * liq;

      for (double flip : {1.0, -1.0}) {
        const IRL::PlanarSeparator a = solveTwoDistances(n0, n1, flip, vf, liq, cell);
        const NewtonResult b = solveTwoDistancesNewton(n0, n1, flip, vf, liq, cell, 0.0,
                                                       m[0] * (liq[0] - cell.calculateCentroid()[0]) +
                                                       m[1] * (liq[1] - cell.calculateCentroid()[1]) +
                                                       m[2] * (liq[2] - cell.calculateCentroid()[2]));
        ++pairs;
        newton_iters += b.iterations;
        if (!b.converged) {
          ++newton_failed;
          ++fail_by_flip[flip > 0 ? 0 : 1];
          ++fail_by_cat[cat];
        }
        ++tot_by_flip[flip > 0 ? 0 : 1];
        ++tot_by_cat[cat];
        if (b.converged && a.getNumberOfPlanes() == 2 && b.separator.getNumberOfPlanes() == 2) {
          dist_diff.push_back(std::max(std::abs(a[0].distance() - b.separator[0].distance()),
                                       std::abs(a[1].distance() - b.separator[1].distance())));
        }
        for (int which = 0; which < 2; ++which) {
          const IRL::PlanarSeparator& sep = which == 0 ? a : b.separator;
          const auto mom = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep);
          const double rvf = std::abs(mom.volume() / cell.calculateVolume() - vf);
          const double rcm = std::abs((mom.volume() > 1.0e-14 ? m * mom.centroid() : target) - target);
          if (which == 0) { res_vf_br.push_back(rvf); res_cm_br.push_back(rcm); }
          else if (b.converged) { res_vf_nw.push_back(rvf); res_cm_nw.push_back(rcm); }
          else { res_vf_bad.push_back(rvf); res_cm_bad.push_back(rcm); }
        }
      }
    }
  }

  auto report = [](const char* name, std::vector<double> v) {
    if (v.empty()) { std::printf("%-26s (none)\n", name); return; }
    std::sort(v.begin(), v.end());
    std::printf("%-26s median %.3e   p99 %.3e   max %.3e\n", name,
                v[v.size() / 2], v[std::min(v.size() - 1, v.size() * 99 / 100)], v.back());
  };
  std::printf("solver validation: %ld solves over %zu categories\n", pairs, allCategories().size());
  report("|d_newton - d_bracketed|", dist_diff);
  report("bracketed  |VF residual|", res_vf_br);
  report("newton     |VF residual|", res_vf_nw);
  report("bracketed  |centroid res|", res_cm_br);
  report("newton     |centroid res|", res_cm_nw);
  report("newton NOT-CONV |VF res|", res_vf_bad);
  report("newton NOT-CONV |cm res|", res_cm_bad);
  std::printf("newton reported failure on %ld of %ld solves (%.2f%%) -> bracketed fallback\n",
              newton_failed, pairs, pairs ? 100.0 * newton_failed / pairs : 0.0);
  std::printf("  by flip:  unflipped %.1f%%   flipped %.1f%%\n",
              tot_by_flip[0] ? 100.0 * fail_by_flip[0] / tot_by_flip[0] : 0.0,
              tot_by_flip[1] ? 100.0 * fail_by_flip[1] / tot_by_flip[1] : 0.0);
  for (const auto& kv : tot_by_cat)
    std::printf("  %-20s %.1f%%\n", kv.first.c_str(),
                100.0 * fail_by_cat[kv.first] / kv.second);
  std::printf("cuts per solve:  bracketed %.1f   newton %.1f   (newton iterations %.1f)\n",
              g_bracketed_solves ? double(g_bracketed_cuts) / g_bracketed_solves : 0.0,
              pairs ? double(g_newton_cuts) / pairs : 0.0,
              pairs ? double(newton_iters) / pairs : 0.0);
  return 0;
}

// ---------------------------------------------------------------------------
// Self-test: the geometry integrator against IRL on shapes IRL can cut
// exactly, and the separator column logic against IRL's own volumes.
// ---------------------------------------------------------------------------
int selftest() {
  const double flip_err = calibrateFlipConvention(7);
  std::printf("flip convention: %s  (mean |VF - IRL| = %.2e)\n",
              g_flip_convention == FlipConvention::kUnionBelow ? "union of below-half-spaces"
                                                               : "complement of intersection",
              flip_err);

  std::mt19937_64 eng(11);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
      IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
  const Vec3 lo{-0.5, -0.5, -0.5}, hi{0.5, 0.5, 0.5};
  double max_vf_err = 0.0, max_c_err = 0.0, max_sep_err = 0.0, max_sd = 0.0;
  for (int s = 0; s < 200; ++s) {
    // Random slab (two planes, liquid between) = unflipped two-plane separator.
    Vec3 n1 = normalized({U(eng), U(eng), U(eng)});
    Vec3 n2 = normalized({U(eng), U(eng), U(eng)});
    if (s % 2 == 0) n2 = -1.0 * n1;   // parallel faces half the time
    const double d1 = 0.3 * U(eng), d2 = 0.3 * U(eng) + (s % 2 == 0 ? -d1 + 0.01 + 0.3 * std::abs(U(eng)) : 0.0);
    Region reg;
    reg.root = reg.opAnd(reg.prim(Quadric::plane(n1, d1)), reg.prim(Quadric::plane(n2, d2)));
    const IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(IRL::Normal(n1[0], n1[1], n1[2]), d1), IRL::Plane(IRL::Normal(n2[0], n2[1], n2[2]), d2), 1.0);
    const auto m = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep);
    const double vf_irl = m.volume();
    const CellMoments cm = integrateCell(reg, lo, hi, 48);
    max_vf_err = std::max(max_vf_err, std::abs(cm.vf - vf_irl));
    if (vf_irl > 1.0e-3) {
      const IRL::Pt c = m.centroid();
      max_c_err = std::max(max_c_err, norm(cm.liq - Vec3{c[0], c[1], c[2]}));
    }
    const CellComparison cmp = compareCell(reg, sep, lo, hi, 48);
    max_sep_err = std::max(max_sep_err, std::abs(cmp.recon_vf - vf_irl));
    max_sd = std::max(max_sd, cmp.symdiff);
  }
  std::printf("slab vs IRL (200 cells, nq=48): max |VF err| %.2e, max centroid err %.2e\n", max_vf_err, max_c_err);
  std::printf("separator columns vs IRL:       max |VF err| %.2e\n", max_sep_err);
  std::printf("self symmetric difference:      max %.2e  (should be ~0)\n", max_sd);

  // Curved: a sphere cap against IRL's paraboloid-free route is not
  // available, so check volume convergence under column refinement instead.
  Region sph;
  sph.root = sph.prim(Quadric::sphere({0.3, -0.2, 0.9}, 1.0));
  const double v32 = integrateCell(sph, lo, hi, 32).vf, v128 = integrateCell(sph, lo, hi, 128).vf;
  std::printf("sphere cap VF nq=32 vs nq=128:  %.3e\n", std::abs(v32 - v128));
  const auto patches = surfacePatches(sph, lo, hi, 64);
  std::printf("sphere cap patch area (nq=64):  %.5f\n", patches[0].area);

  const bool ok = max_vf_err < 1e-3 && max_sep_err < 1e-3 && max_sd < 1e-3 && flip_err < 1e-3;
  std::printf("%s\n", ok ? "SELFTEST PASS" : "SELFTEST FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Summary.
// ---------------------------------------------------------------------------
struct Acc {
  std::vector<double> sd, ang, cent;
  long n = 0, count_ok = 0, routed = 0, sd_bad = 0, vf_bad = 0;
};

double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(p * (v.size() - 1) + 0.5));
  return v[i];
}

void summarize(const std::vector<Row>& rows, const std::vector<std::string>& method_order) {
  std::map<std::string, std::map<std::string, Acc>> acc;
  std::vector<std::string> cats;
  for (const auto& r : rows) {
    if (acc.find(r.cat) == acc.end()) cats.push_back(r.cat);
    Acc& a = acc[r.cat][r.method];
    a.sd.push_back(r.symdiff);
    a.ang.push_back(r.ang_mean);
    a.cent.push_back(r.cent_err);
    ++a.n;
    if (r.n_pred == r.n_true) ++a.count_ok;
    if (r.routed == 1) ++a.routed;
    if (r.symdiff > 0.05) ++a.sd_bad;
    if (r.vf_err > 1.0e-6) ++a.vf_bad;
  }
  std::printf("\n%-19s %-16s %6s | %9s %9s %9s %7s | %7s %7s %7s | %6s %6s %6s\n", "category", "method", "n",
              "sd_mean", "sd_med", "sd_p90", "sd>5%", "ang_med", "ang_p90", "ang_mn", "cnt_ok", "->r2p", "vfbad");
  for (const auto& c : cats) {
    for (const auto& m : method_order) {
      auto it = acc[c].find(m);
      if (it == acc[c].end()) continue;
      const Acc& a = it->second;
      double mean = 0.0, amean = 0.0;
      for (double x : a.sd) mean += x;
      for (double x : a.ang) amean += x;
      mean /= a.n;
      amean /= a.n;
      std::printf("%-19s %-16s %6ld | %9.2e %9.2e %9.2e %6.1f%% | %7.2f %7.2f %7.2f | %5.1f%% %5.1f%% %5.1f%%\n",
                  c.c_str(), m.c_str(), a.n, mean, pct(a.sd, 0.5), pct(a.sd, 0.9),
                  100.0 * a.sd_bad / a.n, pct(a.ang, 0.5), pct(a.ang, 0.9), amean,
                  100.0 * a.count_ok / a.n, 100.0 * a.routed / a.n, 100.0 * a.vf_bad / a.n);
    }
  }
  std::printf("\nsd = |truth XOR recon| / cell volume in the centre cell; ang = area-weighted normal error of\n"
              "true faces (deg); cnt_ok = predicted plane count equals true face count; ->r2p = routed to R2P-Net\n"
              "by the classifier (pipeline methods only).\n");
}

std::vector<Row> readCsv(const std::string& file) {
  std::vector<Row> rows;
  std::ifstream in(file);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string f[22];
    for (auto& x : f) std::getline(ss, x, ',');
    Row r;
    r.cat = f[0]; r.sample = std::stol(f[1]); r.method = f[2];
    r.thickness = std::stod(f[3]); r.r1 = std::stod(f[4]); r.r2 = std::stod(f[5]); r.param = std::stod(f[6]);
    r.vf = std::stod(f[7]); r.n_true = std::stoi(f[8]); r.n_pred = std::stoi(f[9]);
    r.area1 = std::stod(f[10]); r.area2 = std::stod(f[11]); r.symdiff = std::stod(f[12]);
    r.cent_err = std::stod(f[13]); r.ang_mean = std::stod(f[14]); r.ang_max = std::stod(f[15]);
    r.cls = std::stoi(f[16]); r.routed = std::stoi(f[17]); r.aux1 = std::stod(f[18]);
    r.aux2 = std::stod(f[19]); r.vf_err = std::stod(f[20]); r.ms = std::stod(f[21]);
    rows.push_back(r);
  }
  return rows;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::vector<std::string> cats = allCategories();
  std::vector<std::string> methods = {"plicnet", "r2pnet", "pipeline_nopass", "pipeline", "oracle"};
  const std::vector<std::string> method_order = {"oracle_cm", "oracle", "plicnet", "r2p_mof", "r2pnet", "r2pnet_cm",
                                                  "pipeline_nopass", "pipeline_nopass_cm", "pipeline",
                                                  "pipeline_cm", "r2pnet_nw", "pipeline_nw",
                                                  "r2pnet_nw_film", "r2pnet_nw_anti",
                                                  "pipeline_nw_film", "pipeline_nw_anti",
                                                  "r2pnet_nw_pca", "pipeline_nw_pca",
                                                  "r2pnet_nw_vfsum", "pipeline_nw_vfsum"};
  long n = 1000;
  unsigned long long seed = 1;
  double noise = 0.0;
  int nq = 32;
  std::string out = "r2pnet_bench.csv";
  bool invert = false;   // swap phases: every liquid film becomes a gas film

  for (int a = 1; a < argc; ++a) {
    const std::string k = argv[a];
    auto next = [&]() -> std::string {
      if (a + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(2); }
      return argv[++a];
    };
    if (k == "--selftest") return selftest();
    if (k == "--shift-probe") {
      // Does IRL's setDistanceToMatchVolumeFraction move both planes by the
      // SAME shift? solveTwoPlaneDistances assumes so when it recovers s.
      const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
      const IRL::Pt cc = cell.calculateCentroid();
      std::mt19937_64 eng(5);
      std::uniform_real_distribution<double> U(-1.0, 1.0);
      for (int t = 0; t < 5; ++t) {
        IRL::Normal n0(U(eng), U(eng), U(eng));
        n0.normalize();
        IRL::Normal n1 = -n0;
        n1[0] += 0.2 * U(eng);
        n1.normalize();
        const double c0 = n0 * cc, c1 = n1 * cc, g = 0.1 * U(eng), vf = 0.3;
        IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
            IRL::Plane(n0, c0 + g), IRL::Plane(n1, c1 - g), 1.0);
        IRL::setDistanceToMatchVolumeFraction(cell, vf, &sep, 1.0e-12);
        const double s0 = sep[0].distance() - c0 - g, s1 = sep[1].distance() - c1 + g;
        const double vf_out =
            IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep).volume() /
            cell.calculateVolume();
        std::printf("planes=%d  s from plane0 %+.6f  from plane1 %+.6f  diff %.2e   VF %.6f\n",
                    (int)sep.getNumberOfPlanes(), s0, s1, std::abs(s0 - s1), vf_out);
      }
      return 0;
    }
    if (k == "--poly-probe") {
      // IRL's plane polygon vs the locally computed one (verified against
      // finite differences), for two-plane separators in both flip states.
      const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
      const IRL::Pt lo(-0.5, -0.5, -0.5), hi(0.5, 0.5, 0.5);
      std::mt19937_64 eng(5);
      std::uniform_real_distribution<double> U(-1.0, 1.0);
      for (int t = 0; t < 4; ++t) {
        IRL::Normal n0(U(eng), U(eng), U(eng));
        n0.normalize();
        IRL::Normal n1 = -n0;
        n1[0] += 0.2 * U(eng); n1[1] += 0.2 * U(eng);
        n1.normalize();
        for (double flip : {1.0, -1.0}) {
          // flip=+1: liquid slab between the planes. flip=-1: gas film between
          // them, i.e. a genuine flipped configuration rather than a
          // degenerate one.
          const double d = flip > 0 ? 0.15 : -0.1;
          const IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
              IRL::Plane(n0, d), IRL::Plane(n1, d), flip);
          const CutState mine = evaluateCut(sep, cell, lo, hi);
          double irl_area[2] = {0.0, 0.0};
          for (int p = 0; p < 2; ++p) {
            const IRL::Polygon poly =
                IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, sep, sep[p]);
            if (poly.getNumberOfVertices() >= 3)
              irl_area[p] = std::abs(IRL::getVolumeMoments<IRL::VolumeMoments>(poly).volume());
          }
          const double vf = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep)
                                .volume() / cell.calculateVolume();
          std::printf("flip %+g VF %.3f  mine A=(%.4f %.4f)   IRL A=(%.4f %.4f)\n", flip, vf,
                      mine.area[0], mine.area[1], irl_area[0], irl_area[1]);
        }
      }
      return 0;
    }
    if (k == "--flip-probe") {
      // Does the liquid volume grow as both planes move outward, in BOTH
      // flip states? The distance solve assumes it does.
      const IRL::RectangularCuboid cell = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt(-0.5, -0.5, -0.5), IRL::Pt(0.5, 0.5, 0.5));
      const IRL::Normal n0(0.0, 0.0, 1.0), n1(0.0, 0.0, -1.0);
      for (double flip : {1.0, -1.0}) {
        std::printf("flip %+g :", flip);
        for (double t : {-0.6, -0.3, 0.0, 0.3, 0.6}) {
          const IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
              IRL::Plane(n0, t), IRL::Plane(n1, t), flip);
          const double vf =
              IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, sep).volume() /
              cell.calculateVolume();
          std::printf("  s=%+.1f VF=%.4f", t, vf);
        }
        std::printf("\n");
      }
      return 0;
    }
    if (k == "--jacobian-check") return jacobianCheck(2000, 3);
    if (k == "--validate-solver") {
      long vn = 200;
      double vnoise = 0.0;
      unsigned long long vseed = 5;
      for (int b = a + 1; b + 1 < argc; b += 2) {
        const std::string kk = argv[b];
        if (kk == "--n") vn = std::stol(argv[b + 1]);
        else if (kk == "--noise") vnoise = std::stod(argv[b + 1]);
        else if (kk == "--seed") vseed = std::stoull(argv[b + 1]);
      }
      return validateSolver(vn, vseed, vnoise);
    }
    if (k == "--summarize") {
      std::vector<Row> rows;
      for (++a; a < argc; ++a) { auto r = readCsv(argv[a]); rows.insert(rows.end(), r.begin(), r.end()); }
      summarize(rows, method_order);
      return 0;
    }
    if (k == "--cat") { const std::string v = next(); if (v != "all") cats = split(v); }
    else if (k == "--n") n = std::stol(next());
    else if (k == "--seed") seed = std::stoull(next());
    else if (k == "--noise") noise = std::stod(next());
    else if (k == "--methods") methods = split(next());
    else if (k == "--nq") nq = std::stoi(next());
    else if (k == "--out") out = next();
    else if (k == "--invert") invert = true;
    else if (k == "--liquid-centroid") bench::g_film_centroid = false;
    else { std::cerr << "unknown argument " << k << "\n"; return 2; }
  }

  calibrateFlipConvention(7);
  SceneSampler sampler(seed);
  std::ofstream csv(out);
  csv << kHeader << '\n';
  csv.precision(8);
  std::vector<Row> rows;
  const Vec3 lo{-0.5, -0.5, -0.5}, hi{0.5, 0.5, 0.5};
  const int nq_metric = 2 * nq;

  for (const auto& cat : cats) {
    const auto t_cat = std::chrono::steady_clock::now();
    long rejected = 0;
    for (long s = 0; s < n; ++s) {
      Scene scene = sampler.sample(cat);
      if (invert) scene.region.root = scene.region.opNot(scene.region.root);
      Block blk;
      fillBlock(scene.region, nq, &blk);
      if (!blk.mixed(Block::kC, Block::kC, Block::kC)) { ++rejected; --s; continue; }

      // Truth in the centre cell, from the clean geometry and before noise.
      std::vector<TruthFace> faces;
      for (const auto& p : surfacePatches(scene.region, lo, hi, nq_metric))
        if (p.area > 0.0) faces.push_back({p.normal(), p.area});
      std::sort(faces.begin(), faces.end(), [](const TruthFace& x, const TruthFace& y) { return x.area > y.area; });
      const IRL::Pt true_liq = blk.liq(Block::kC, Block::kC, Block::kC);

      perturbBlock(noise, sampler.engine(), &blk);

      Row base;
      base.cat = cat;
      base.sample = s;
      base.thickness = scene.thickness;
      base.r1 = scene.radius1;
      base.r2 = scene.radius2;
      base.param = scene.param;
      base.vf = blk.vf(Block::kC, Block::kC, Block::kC);
      for (const auto& f : faces) if (f.area > kFaceArea) ++base.n_true;
      base.area1 = faces.size() > 0 ? faces[0].area : 0.0;
      base.area2 = faces.size() > 1 ? faces[1].area : 0.0;

      double vf_sum = 0.0;
      for (int ii = 0; ii < 3; ++ii)
        for (int jj = 0; jj < 3; ++jj)
          for (int kk = 0; kk < 3; ++kk) vf_sum += blk.vf(ii, jj, kk);
      base.rule_flip = vf_sum >= 0.5 * 27.0 ? 1 : 0;
      bench::g_force_flip = 2;
      base.pca_flip = flipFor(blk, Block::kC, Block::kC, Block::kC) ? 1 : 0;
      bench::g_force_flip = -1;

      for (const auto& method_name : methods) {
        Row row = base;
        row.method = method_name;
        // _film / _anti: phase 0 forced to the liquid (the film, in every film
        // scene) or to the gas, instead of the deployed 3^3 VF-sum rule.
        std::string m = method_name;
        bench::g_force_flip = -1;
        for (const auto& [suffix, force] : {std::pair<std::string, int>{"_film", 0}, {"_anti", 1}, {"_pca", 2}, {"_vfsum", 3}}) {
          if (m.size() > suffix.size() && m.compare(m.size() - suffix.size(), suffix.size(), suffix) == 0) {
            m = m.substr(0, m.size() - suffix.size());
            bench::g_force_flip = force;
          }
        }
        const auto t0 = std::chrono::steady_clock::now();
        IRL::PlanarSeparator sep;
        const int C = Block::kC;
        if (m == "plicnet") {
          sep = reconstructPlicnet(blk, C, C, C);
        } else if (m == "r2pnet") {
          R2PNetOutput raw;
          sep = reconstructR2PNet(blk, C, C, C, &raw);
          row.aux1 = raw.n1.calculateMagnitude();
          row.aux2 = raw.n2.calculateMagnitude();
        } else if (m == "pipeline" || m == "pipeline_nopass") {
          const PipelineResult pr = reconstructPipeline(blk, m == "pipeline");
          sep = pr.centre;
          row.cls = pr.centre_class;
          row.routed = pr.centre_routed_r2p ? 1 : 0;
        } else if (m == "oracle") {
          sep = oracle(blk, scene.region, faces, nq_metric);
        } else if (m == "r2pnet_cm" || m == "r2pnet_nw") {
          bench::g_solver = m == "r2pnet_nw" ? bench::SolverKind::kNewton
                                             : bench::SolverKind::kBracketed;
          R2PNetOutput raw;
          sep = reconstructR2PNet(blk, C, C, C, &raw);
          bench::g_solver = bench::SolverKind::kLegacy;
          row.aux1 = raw.n1.calculateMagnitude();
          row.aux2 = raw.n2.calculateMagnitude();
          row.cls = classifyCell(blk, C, C, C);
        } else if (m == "pipeline_cm" || m == "pipeline_nopass_cm" ||
                   m == "pipeline_nw" || m == "pipeline_nopass_nw") {
          const bool newton = m.size() > 3 && m.substr(m.size() - 3) == "_nw";
          bench::g_solver = newton ? bench::SolverKind::kNewton : bench::SolverKind::kBracketed;
          const PipelineResult pr = reconstructPipeline(blk, m == "pipeline_cm" || m == "pipeline_nw");
          bench::g_solver = bench::SolverKind::kLegacy;
          sep = pr.centre;
          row.cls = pr.centre_class;
          row.routed = pr.centre_routed_r2p ? 1 : 0;
        } else if (m == "oracle_cm") {
          sep = oracleCM(blk, scene.region, faces, nq_metric);
        } else if (m == "r2p_mof") {
          sep = reconstructR2PMOF(blk, C, C, C);
        } else {
          std::cerr << "unknown method " << m << "\n";
          return 2;
        }
        bench::g_force_flip = -1;
        row.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        score(scene.region, faces, true_liq, sep, nq_metric, &row);
        writeRow(csv, row);
        rows.push_back(row);
      }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cat).count();
    std::fprintf(stderr, "# %-19s %ld samples in %.1fs (%ld resampled)\n", cat.c_str(), n, secs, rejected);
  }
  if (g_legacy_solves > 0) {
    std::fprintf(stderr, "# legacy solver: %ld solves, %.1f cuts/solve\n", g_legacy_solves,
                 double(g_legacy_cuts) / g_legacy_solves);
  }
  if (g_bracketed_solves > 0) {
    std::fprintf(stderr, "# bracketed solver: %ld solves, %.1f cuts/solve\n", g_bracketed_solves,
                 double(g_bracketed_cuts) / g_bracketed_solves);
  }
  if (g_newton_solves > 0) {
    static const char* kReason[7] = {"none", "no-area", "tiny-volume", "singular",
                                     "stalled", "max-iter", "one-plane"};
    std::fprintf(stderr, "# newton one-plane reductions: %ld\n", g_newton_one_plane);
    std::fprintf(stderr, "# newton failures by reason:");
    for (int i = 1; i < 7; ++i)
      if (g_newton_fail_reason[i] > 0)
        std::fprintf(stderr, " %s=%ld", kReason[i], g_newton_fail_reason[i]);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr,
                 "# newton solver: %ld solves, %.2f%% fell back to bracketed, %.1f cuts/solve\n",
                 g_newton_solves, 100.0 * g_newton_fallbacks / g_newton_solves,
                 double(g_newton_cuts) / g_newton_solves);
  }
  summarize(rows, method_order);
  return 0;
}
