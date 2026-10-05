// Reconstruction methods evaluated by the benchmark.
//
// The per-cell logic here is a faithful copy of R2P3D_Net::getReconstruction
// in examples/new_advector/reconstruction_types.cpp (PLICNet branch, R2P-Net
// branch with its one-plane fallback, classifier routing), with two
// deliberate differences:
//
//   * The reconstructionWithR2P3D call that seeds the R2P branch is skipped.
//     For interior cells its result is always overwritten, so the output is
//     unchanged; it would only need advected surface elements the benchmark
//     does not have.
//   * Nothing here writes the solver's diagnostic Data fields.
//
// The paraboloid pass is not copied: r2ppass::run from
// examples/new_advector/r2p_paraboloid_pass.h is called as-is.
//
// If the logic in reconstruction_types.cpp changes, this file must be
// updated to match, or the benchmark stops measuring the deployed method.

#ifndef EXAMPLES_R2PNET_BENCH_METHODS_H_
#define EXAMPLES_R2PNET_BENCH_METHODS_H_

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/interface_reconstruction_methods/plane_distance.h"
#include "irl/interface_reconstruction_methods/r2p_neighborhood.h"
#include "irl/interface_reconstruction_methods/r2p_optimization.h"
#include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
#include "irl/interface_reconstruction_methods/reconstruction_interface.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"
#include "examples/new_advector/ml_classifier.h"
#include "examples/new_advector/plicnet.h"
#include "examples/new_advector/r2pnet.h"
#include "examples/new_advector/r2p_newton_distance.h"
#include "examples/new_advector/r2p_paraboloid_pass.h"

#include "examples/r2pnet_bench/distance_solve.h"
#include "examples/r2pnet_bench/geometry.h"
#include "examples/r2pnet_bench/newton_solve.h"

namespace bench {
// Which distance solve every R2PDistanceSolver call uses -- including the
// calls inside r2ppass.
//   kLegacy    the deployed solver: one common shift, volume only
//   kBracketed volume + centroid, nested Illinois (the reference)
//   kNewton    volume + centroid, damped Newton with exact derivatives
enum class SolverKind { kLegacy, kBracketed, kNewton };
inline SolverKind g_solver = SolverKind::kLegacy;
// Phase-0 choice for the R2P-Net branch and the classifier:
//   -1 the deployed rule (r2pFlip in reconstruction_types.cpp: PCA, below)
//    0 liquid, 1 gas (forced)
//    2 PCA rule: the phase whose centroid cloud is flatter
//    3 the old VF-sum rule (3^3 VF sum >= 13.5)
inline int g_force_flip = -1;
// When true, flipped cells hand the distance solve a liquid centroid rebuilt
// from the GAS centroid, so the solve matches the film phase's own moment
// instead of recovering it from the liquid centroid (noise amplified by
// vf/(1-vf) for a thin gas film).
inline bool g_film_centroid = true;   // deployed behaviour
// Cell cuts spent by the deployed solver, for cost comparison.
inline long g_legacy_cuts = 0;
inline long g_legacy_solves = 0;
}  // namespace bench

// ---------------------------------------------------------------------------
// Verbatim copy of R2PDistanceSolver from reconstruction_types.cpp. It has
// to live at global scope with this exact signature, because
// r2p_paraboloid_pass.h declares it and calls it.
// ---------------------------------------------------------------------------
inline void R2PDistanceSolverImpl(double VF_target, IRL::Pt bary_target,
                                  IRL::PlanarSeparator& a_interface,
                                  IRL::RectangularCuboid cell) {
  IRL::Pt cell_centroid = cell.calculateCentroid();
  int sign = a_interface.isNotFlipped() ? 1 : -1;

  if (sign == -1) {
    bary_target = (cell_centroid - bary_target * VF_target);
    bary_target[0] = bary_target[0] / (1 - VF_target);
    bary_target[1] = bary_target[1] / (1 - VF_target);
    bary_target[2] = bary_target[2] / (1 - VF_target);
  }

  IRL::Normal n = a_interface[0].normal() - a_interface[1].normal();
  if (n.calculateMagnitude() < 1.0e-12) {
    n = a_interface[0].normal();
  }
  n.normalize();

  double t = IRL::dotProduct(bary_target, n) - IRL::dotProduct(cell_centroid, n);
  double dist1 = IRL::dotProduct(cell_centroid, a_interface[0].normal()) + t;
  double dist2 = IRL::dotProduct(cell_centroid, a_interface[1].normal()) - t;

  double side = (cell.calculateSideLength(0) + cell.calculateSideLength(1) +
                 cell.calculateSideLength(2)) / 3.0;
  double tol = 1e-14;
  IRL::Pt bary;

  {
    int max_iter = 200;
    int iter = 0;
    double VF_cut = 0.0;
    double error = 1.0;

    ++bench::g_legacy_solves;
    auto setInterval = [&](double shift) {
      ++bench::g_legacy_cuts;
      a_interface[0] = IRL::Plane(a_interface[0].normal(), dist1 + sign * shift);
      a_interface[1] = IRL::Plane(a_interface[1].normal(), dist2 + sign * shift);
      auto m = IRL::getNormalizedVolumeMoments<IRL::VolumeMoments>(cell, a_interface);
      bary = m.volume() > 1.0e-14 * cell.calculateVolume() ? m.centroid() : cell_centroid;
      return m.volume() / cell.calculateVolume();
    };

    double VF_zero = setInterval(0.0);
    double interval_min = 0.0;
    double interval_max = sign * VF_zero > sign * VF_target ? -0.25 * side : 0.25 * side;
    double VF_bound = setInterval(interval_max);

    while (iter < max_iter && (VF_zero - VF_target) * (VF_bound - VF_target) > 0.0) {
      interval_min = interval_max;
      interval_max *= 2.0;
      VF_bound = setInterval(interval_max);
      ++iter;
      if (std::abs(interval_max) > 20.0 * side) break;
    }
    if (interval_max < interval_min) {
      std::swap(interval_min, interval_max);
    }

    std::array<double, 3> bounding_values{{interval_min, 0.5 * (interval_min + interval_max), interval_max}};

    VF_cut = setInterval(bounding_values[1]);
    error = std::abs(VF_cut - VF_target);

    iter = 0;
    while (error > tol && iter < max_iter) {
      if (sign * VF_cut < sign * VF_target) {
        bounding_values[0] = bounding_values[1];
      } else {
        bounding_values[2] = bounding_values[1];
      }
      bounding_values[1] = 0.5 * (bounding_values[0] + bounding_values[2]);
      VF_cut = setInterval(bounding_values[1]);
      error = std::abs(VF_cut - VF_target);
      ++iter;
    }
    IRL::cleanReconstruction(cell, VF_target, &a_interface);
  }
}

void R2PDistanceSolver(double VF_target, IRL::Pt bary_target,
                       IRL::PlanarSeparator& a_interface,
                       IRL::RectangularCuboid cell) {
  switch (bench::g_solver) {
    case bench::SolverKind::kBracketed:
      bench::momentDistanceSolver(VF_target, bary_target, a_interface, cell);
      break;
    case bench::SolverKind::kNewton:
      // The ported production version in examples/new_advector, so the
      // benchmark measures exactly what the advector would run.
      r2pnewton::R2PNewtonDistanceSolver(VF_target, bary_target, a_interface, cell);
      break;
    default:
      R2PDistanceSolverImpl(VF_target, bary_target, a_interface, cell);
      break;
  }
}

namespace bench {

// ---------------------------------------------------------------------------
// A 9x9x9 block of cells: 3^3 interior (indices 0..2, centre (1,1,1) at the
// origin) plus 3 ghost layers. That is exactly enough for the full pipeline:
// the paraboloid pass on the centre needs reconstructions in -1..3, whose
// classifier stencils reach -3..5.
// ---------------------------------------------------------------------------
struct Block {
  BasicMesh mesh;
  Data<double> vf;
  Data<IRL::Pt> liq;
  Data<IRL::Pt> gas;

  Block() : mesh(3, 3, 3, 3) {
    mesh.setCellBoundaries(IRL::Pt(-1.5, -1.5, -1.5), IRL::Pt(1.5, 1.5, 1.5));
    vf = Data<double>(&mesh);
    liq = Data<IRL::Pt>(&mesh);
    gas = Data<IRL::Pt>(&mesh);
  }
  static constexpr int kC = 1;   // centre index

  IRL::RectangularCuboid cell(int i, int j, int k) const {
    return IRL::RectangularCuboid::fromBoundingPts(
        IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
        IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
  }
  bool mixed(int i, int j, int k) const {
    return vf(i, j, k) >= IRL::global_constants::VF_LOW &&
           vf(i, j, k) <= IRL::global_constants::VF_HIGH;
  }
};

// Fills the block from a region. nq = columns per cell edge.
inline void fillBlock(const Region& reg, int nq, Block* blk) {
  const BasicMesh& m = blk->mesh;
  for (int i = m.imino(); i <= m.imaxo(); ++i)
    for (int j = m.jmino(); j <= m.jmaxo(); ++j)
      for (int k = m.kmino(); k <= m.kmaxo(); ++k) {
        const Vec3 lo{m.x(i), m.y(j), m.z(k)};
        const Vec3 hi{m.x(i + 1), m.y(j + 1), m.z(k + 1)};
        const CellMoments cm = integrateCell(reg, lo, hi, nq);
        blk->vf(i, j, k) = cm.vf;
        blk->liq(i, j, k) = IRL::Pt(cm.liq[0], cm.liq[1], cm.liq[2]);
        blk->gas(i, j, k) = IRL::Pt(cm.gas[0], cm.gas[1], cm.gas[2]);
      }
}

// Barycentre noise on interfacial cells, mirroring data_gen::perturbMoments:
// cell-relative components get N(0, sigma) and are clipped to +/-0.5.
template <class Engine>
inline void perturbBlock(double sigma, Engine& eng, Block* blk) {
  if (sigma <= 0.0) return;
  std::normal_distribution<double> noise(0.0, sigma);
  const BasicMesh& m = blk->mesh;
  for (int i = m.imino(); i <= m.imaxo(); ++i)
    for (int j = m.jmino(); j <= m.jmaxo(); ++j)
      for (int k = m.kmino(); k <= m.kmaxo(); ++k) {
        const double v = blk->vf(i, j, k);
        if (v <= IRL::global_constants::VF_LOW || v >= IRL::global_constants::VF_HIGH) continue;
        const IRL::Pt cc(m.xm(i), m.ym(j), m.zm(k));
        for (IRL::Pt* p : {&blk->liq(i, j, k), &blk->gas(i, j, k)}) {
          for (int d = 0; d < 3; ++d) {
            const double rel = (*p)[d] - cc[d] + noise(eng);
            (*p)[d] = cc[d] + std::min(0.5, std::max(-0.5, rel));
          }
        }
      }
}

// ---------------------------------------------------------------------------
// Pieces copied from R2P3D_Net.
// ---------------------------------------------------------------------------
inline IRL::Normal PCA_Normal(const std::vector<IRL::Pt>& points) {
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto& pt : points) centroid += Eigen::Vector3d(pt[0], pt[1], pt[2]);
  centroid = centroid / double(points.size());
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& pt : points) {
    Eigen::Vector3d d = Eigen::Vector3d(pt[0], pt[1], pt[2]) - centroid;
    covariance += d * d.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
  Eigen::Vector3d local_z = eigensolver.eigenvectors().col(0).normalized();
  IRL::Normal n = IRL::Normal(local_z[0], local_z[1], local_z[2]);
  n.normalize();
  return n;
}

// Sphericity (smallest / largest covariance eigenvalue) of the centroids of
// the 3^3 cells holding some of the given phase. A film phase gives a flat
// layer of points; the phase around it fills the stencil on both sides.
// Returns -1 with fewer than 3 points.
inline double phaseSphericity(const Block& b, int i, int j, int k, bool gas) {
  std::vector<Eigen::Vector3d> pts;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        const double f = gas ? 1.0 - b.vf(ii, jj, kk) : b.vf(ii, jj, kk);
        if (f <= IRL::global_constants::VF_LOW) continue;
        const IRL::Pt& p = gas ? b.gas(ii, jj, kk) : b.liq(ii, jj, kk);
        pts.emplace_back(p[0], p[1], p[2]);
      }
  if (pts.size() < 3) return -1.0;
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (const auto& p : pts) c += p;
  c /= double(pts.size());
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto& p : pts) cov += (p - c) * (p - c).transpose();
  const Eigen::Vector3d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(cov).eigenvalues();
  return ev(2) > 1.0e-30 ? std::max(0.0, ev(0)) / ev(2) : -1.0;
}

inline bool flipFor(const Block& b, int i, int j, int k) {
  if (g_force_flip == 0 || g_force_flip == 1) return g_force_flip == 1;
  double vol = 0.0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) vol += b.vf(ii, jj, kk);
  const bool rule = vol >= 0.5 * 27.0;
  if (g_force_flip == 3) return rule;
  const double s_liq = phaseSphericity(b, i, j, k, false);
  const double s_gas = phaseSphericity(b, i, j, k, true);
  if (s_liq < 0.0 || s_gas < 0.0) return rule;
  return s_gas < s_liq;
}

inline void undoPlicnetFrame(int dir1, int dir2, IRL::Normal* normal_ptr) {
  IRL::Normal& normal = *normal_ptr;
  double temp;
  switch (dir2) {
    case 1: temp = normal[0]; normal[0] = normal[1]; normal[1] = temp; break;
    case 2: temp = normal[1]; normal[1] = normal[2]; normal[2] = temp; break;
    case 3: temp = normal[0]; normal[0] = normal[2]; normal[2] = temp; break;
    case 4: temp = normal[1]; normal[1] = normal[2]; normal[2] = temp;
            temp = normal[0]; normal[0] = normal[1]; normal[1] = temp; break;
    case 5: temp = normal[0]; normal[0] = normal[2]; normal[2] = temp;
            temp = normal[0]; normal[0] = normal[1]; normal[1] = temp; break;
  }
  switch (dir1) {
    case 1: normal[0] = -normal[0]; break;
    case 2: normal[1] = -normal[1]; break;
    case 3: normal[2] = -normal[2]; break;
    case 4: normal[0] = -normal[0]; normal[1] = -normal[1]; break;
    case 5: normal[0] = -normal[0]; normal[2] = -normal[2]; break;
    case 6: normal[1] = -normal[1]; normal[2] = -normal[2]; break;
    case 7: normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; break;
  }
}

// The 189 cell-relative moments of the 3^3 stencil about (i,j,k), phase
// swapped when `flip`, plus the VF-weighted centre used for reflection.
inline void stencilMoments(const Block& b, int i, int j, int k, bool flip,
                           double (&moments)[189], double (&center)[3]) {
  const BasicMesh& mesh = b.mesh;
  double m000 = 0, m100 = 0, m010 = 0, m001 = 0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        const int idx = 7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
        const IRL::Pt& first = flip ? b.gas(ii, jj, kk) : b.liq(ii, jj, kk);
        const IRL::Pt& second = flip ? b.liq(ii, jj, kk) : b.gas(ii, jj, kk);
        moments[idx] = flip ? 1.0 - b.vf(ii, jj, kk) : b.vf(ii, jj, kk);
        moments[idx + 1] = (first[0] - mesh.xm(ii)) / mesh.dx();
        moments[idx + 2] = (first[1] - mesh.ym(jj)) / mesh.dy();
        moments[idx + 3] = (first[2] - mesh.zm(kk)) / mesh.dz();
        moments[idx + 4] = (second[0] - mesh.xm(ii)) / mesh.dx();
        moments[idx + 5] = (second[1] - mesh.ym(jj)) / mesh.dy();
        moments[idx + 6] = (second[2] - mesh.zm(kk)) / mesh.dz();
        m000 += moments[idx];
        m100 += (moments[idx + 1] + (ii - i)) * moments[idx];
        m010 += (moments[idx + 2] + (jj - j)) * moments[idx];
        m001 += (moments[idx + 3] + (kk - k)) * moments[idx];
      }
  center[0] = m100 / m000;
  center[1] = m010 / m000;
  center[2] = m001 / m000;
}

inline IRL::Normal plicnetNormal(const Block& b, int i, int j, int k) {
  const BasicMesh& mesh = b.mesh;
  const bool flip_plic = (b.vf(i, j, k) >= 0.5);
  double moments[189] = {0};
  double center[3] = {0};
  stencilMoments(b, i, j, k, flip_plic, moments, center);
  int dir1 = 0, dir2 = 0;
  double n[3] = {0};
  plicnet::reflect_moments(moments, center, &dir1, &dir2);
  plicnet::get_normal(moments, n);
  IRL::Normal normal(n[0], n[1], n[2]);
  undoPlicnetFrame(dir1, dir2, &normal);
  if (!flip_plic) normal = -normal;
  normal[0] *= mesh.dx();
  normal[1] *= mesh.dy();
  normal[2] *= mesh.dz();
  normal.normalize();
  return normal;
}

// PLICNet branch.
inline IRL::PlanarSeparator reconstructPlicnet(const Block& b, int i, int j, int k) {
  const IRL::Normal normal = plicnetNormal(b, i, j, k);
  const double distance = IRL::findDistanceOnePlane(b.cell(i, j, k), b.vf(i, j, k), normal);
  return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal, distance));
}

// Raw network output for the R2P branch, exposed so the benchmark can log it.
struct R2PNetOutput {
  IRL::Normal n1, n2;   // after the frame is undone, before mesh scaling
  bool one_plane = false;
  bool flip = false;
};

// R2P-Net branch (the non-boundary path of R2P3D_Net).
inline IRL::PlanarSeparator reconstructR2PNet(const Block& b, int i, int j, int k,
                                              R2PNetOutput* raw = nullptr) {
  const BasicMesh& mesh = b.mesh;
  double vol = 0.0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) vol += b.vf(ii, jj, kk);
  const bool flip = flipFor(b, i, j, k);

  std::vector<IRL::Pt> points;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        if (!flip) {
          if (b.vf(ii, jj, kk) > IRL::global_constants::VF_LOW) points.push_back(b.liq(ii, jj, kk));
        } else {
          if ((1 - b.vf(ii, jj, kk)) > IRL::global_constants::VF_LOW) points.push_back(b.gas(ii, jj, kk));
        }
      }

  double moments[189] = {0};
  double center_unused[3] = {0};
  stencilMoments(b, i, j, k, flip, moments, center_unused);
  // stencilMoments returns the VF-weighted centre; R2P3D_Net orients the PCA
  // direction against the same quantity (m100/m000, ...), in cell units.
  IRL::Normal dir = PCA_Normal(points);
  const IRL::Pt bary(center_unused[0], center_unused[1], center_unused[2]);
  if (IRL::dotProduct(dir, bary) < 0) dir = -dir;

  double center[3] = {dir[0], dir[1], dir[2]};
  int direction = 0, direction2 = 0;
  r2pnet::reflect_moments(moments, center, &direction, &direction2);

  double temp;
  switch (direction) {
    case 1: center[0] = -center[0]; break;
    case 2: center[1] = -center[1]; break;
    case 3: center[2] = -center[2]; break;
    case 4: center[0] = -center[0]; center[1] = -center[1]; break;
    case 5: center[0] = -center[0]; center[2] = -center[2]; break;
    case 6: center[1] = -center[1]; center[2] = -center[2]; break;
    case 7: center[0] = -center[0]; center[1] = -center[1]; center[2] = -center[2]; break;
  }
  switch (direction2) {
    case 1: temp = center[0]; center[0] = center[1]; center[1] = temp; break;
    case 2: temp = center[1]; center[1] = center[2]; center[2] = temp; break;
    case 3: temp = center[0]; center[0] = center[2]; center[2] = temp; break;
    case 4: temp = center[0]; center[0] = center[1]; center[1] = temp;
            temp = center[1]; center[1] = center[2]; center[2] = temp; break;
    case 5: temp = center[0]; center[0] = center[1]; center[1] = temp;
            temp = center[0]; center[0] = center[2]; center[2] = temp; break;
  }

  double input[192] = {0};
  std::copy(moments, moments + 189, input);
  input[189] = center[0];
  input[190] = center[1];
  input[191] = center[2];

  double n[6] = {0, 0, 0, 0, 0, 0};
  r2pnet::get_normals(input, n);
  IRL::Normal normal1(n[0], n[1], n[2]);
  IRL::Normal normal2(n[3], n[4], n[5]);
  undoPlicnetFrame(direction, direction2, &normal1);
  undoPlicnetFrame(direction, direction2, &normal2);

  const bool one_plane = (normal2.calculateMagnitude() < 0.85 || normal1.calculateMagnitude() < 0.85);
  if (raw != nullptr) {
    raw->n1 = normal1;
    raw->n2 = normal2;
    raw->one_plane = one_plane;
    raw->flip = flip;
  }

  const IRL::RectangularCuboid cube = b.cell(i, j, k);
  if (!one_plane) {
    normal1[0] *= mesh.dx(); normal1[1] *= mesh.dy(); normal1[2] *= mesh.dz();
    normal1.normalize();
    normal2[0] *= mesh.dx(); normal2[1] *= mesh.dy(); normal2[2] *= mesh.dz();
    normal2.normalize();
    const int flip_i = flip ? -1 : 1;
    if (!flip) normal1 = -normal1;
    if (!flip) normal2 = -normal2;
    IRL::PlanarSeparator sep = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(normal1, 0), IRL::Plane(normal2, 0), flip_i);
    IRL::Pt target = b.liq(i, j, k);
    const double vfc = b.vf(i, j, k);
    if (g_film_centroid && flip && vfc > 1.0e-8) {
      const IRL::Pt cc = cube.calculateCentroid();
      const IRL::Pt g = b.gas(i, j, k);
      for (int d = 0; d < 3; ++d) target[d] = (cc[d] - (1.0 - vfc) * g[d]) / vfc;
    }
    ::R2PDistanceSolver(vfc, target, sep, cube);
    return sep;
  }

  const double target_vf = b.vf(i, j, k);
  const IRL::Pt target_liq = b.liq(i, j, k);
  const IRL::Pt target_gas = b.gas(i, j, k);
  const IRL::Pt cell_ctr = cube.calculateCentroid();

  auto build_and_score = [&](IRL::Normal nrm, IRL::PlanarSeparator* out) -> double {
    if (nrm.calculateMagnitude() < 0.5) return DBL_MAX;
    nrm.normalize();
    const double d = IRL::findDistanceOnePlane(cube, target_vf, nrm);
    *out = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(nrm, d));
    auto svm = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(cube, *out);
    const double vf_out = svm[0].volume() / cube.calculateVolume();
    if (std::abs(vf_out - target_vf) > 1.0e-6) return DBL_MAX;
    double err = 0.0;
    if (target_vf > IRL::global_constants::VF_LOW) err += IRL::magnitude(target_liq - svm[0].centroid());
    if (target_vf < IRL::global_constants::VF_HIGH) err += IRL::magnitude(target_gas - svm[1].centroid());
    return err;
  };

  IRL::Normal nn_normal = (normal2.calculateMagnitude() < normal1.calculateMagnitude()) ? normal1 : normal2;
  nn_normal[0] *= mesh.dx(); nn_normal[1] *= mesh.dy(); nn_normal[2] *= mesh.dz();
  if (nn_normal.calculateMagnitude() > 0.0) nn_normal.normalize();
  if (!flip) nn_normal = -nn_normal;
  if (IRL::dotProduct(nn_normal, (target_liq - cell_ctr)) > 0) nn_normal = -nn_normal;

  IRL::PlanarSeparator sep_nn, sep_plic;
  const double err_nn = build_and_score(nn_normal, &sep_nn);
  const double err_plic = build_and_score(plicnetNormal(b, i, j, k), &sep_plic);
  return err_plic < err_nn ? sep_plic : sep_nn;
}

// Classifier class for cell (i,j,k), exactly as R2P3D_Net computes it.
inline int classifyCell(const Block& b, int i, int j, int k) {
  const BasicMesh& mesh = b.mesh;
  double vol = 0.0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) vol += b.vf(ii, jj, kk);
  const bool flip = flipFor(b, i, j, k);

  ml_classifier::Stencil stencil;
  const IRL::Pt cell_center(mesh.xm(i), mesh.ym(j), mesh.zm(k));
  for (int ii = 0; ii < 5; ++ii)
    for (int jj = 0; jj < 5; ++jj)
      for (int kk = 0; kk < 5; ++kk) {
        const int gi = i + ii - 2, gj = j + jj - 2, gk = k + kk - 2;
        double v = b.vf(gi, gj, gk);
        if (flip) v = 1 - v;
        stencil.f(ii, jj, kk) = v;
        IRL::Pt bary = flip ? b.gas(gi, gj, gk) : b.liq(gi, gj, gk);
        bary -= cell_center;
        bary[0] /= mesh.dx();
        bary[1] /= mesh.dy();
        bary[2] /= mesh.dz();
        bary *= v;
        stencil.b(ii, jj, kk, 0) = bary[0];
        stencil.b(ii, jj, kk, 1) = bary[1];
        stencil.b(ii, jj, kk, 2) = bary[2];
      }
  return ml_classifier::get_class(stencil);
}

inline IRL::PlanarSeparator emptyOrFull(double vf) {
  const double distance = std::copysign(1.0, vf - 0.5);
  return IRL::PlanarSeparator::fromOnePlane(IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance));
}

struct PipelineResult {
  IRL::PlanarSeparator centre;
  int centre_class = 0;
  bool centre_routed_r2p = false;
};

// Full deployed pipeline: classify and reconstruct every cell the paraboloid
// pass can read, then run the pass, and return the centre cell.
inline PipelineResult reconstructPipeline(const Block& b, bool run_paraboloid_pass) {
  const BasicMesh& mesh = b.mesh;
  Data<IRL::PlanarSeparator> iface(&mesh);
  PipelineResult res;
  for (int i = -1; i <= 3; ++i)
    for (int j = -1; j <= 3; ++j)
      for (int k = -1; k <= 3; ++k) {
        if (!b.mixed(i, j, k)) {
          iface(i, j, k) = emptyOrFull(b.vf(i, j, k));
          continue;
        }
        const int cls = classifyCell(b, i, j, k);
        const bool use_r2p = (cls == 4 || cls == 6);
        iface(i, j, k) = use_r2p ? reconstructR2PNet(b, i, j, k) : reconstructPlicnet(b, i, j, k);
        if (i == Block::kC && j == Block::kC && k == Block::kC) {
          res.centre_class = cls;
          res.centre_routed_r2p = use_r2p;
        }
      }
  if (run_paraboloid_pass) {
    // run() ends with a periodic updateBorder() that only rewrites ghost
    // cells; the centre is interior and unaffected.
    r2ppass::Options opt;
    r2ppass::run(b.vf, b.liq, b.gas, &iface, opt);
  }
  res.centre = iface(Block::kC, Block::kC, Block::kC);
  return res;
}

// R2P optimizer (IRL) seeded with MOF, using only the 3^3 moments. A
// non-ML reference; the production R2P seeds from advected normals instead.
inline IRL::PlanarSeparator reconstructR2PMOF(const Block& b, int i, int j, int k) {
  IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
  neighborhood.resize(27);
  neighborhood.setCenterOfStencil(13);
  IRL::RectangularCuboid cells[27];
  IRL::SeparatedMoments<IRL::VolumeMoments> sm[27];
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        const int ind = (ii - i + 1) * 9 + (jj - j + 1) * 3 + (kk - k + 1);
        cells[ind] = b.cell(ii, jj, kk);
        const double vol = cells[ind].calculateVolume();
        sm[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
            IRL::VolumeMoments(b.vf(ii, jj, kk) * vol, b.liq(ii, jj, kk)),
            IRL::VolumeMoments((1.0 - b.vf(ii, jj, kk)) * vol, b.gas(ii, jj, kk)));
        neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind), &cells[ind], &sm[ind]);
      }
  const IRL::RectangularCuboid cell = b.cell(i, j, k);
  IRL::PlanarSeparator init = IRL::reconstructionWithMOF3D(cell, sm[13]);
  neighborhood.setSurfaceArea(IRL::getReconstructionSurfaceArea(cell, init));
  return IRL::reconstructionWithR2P3D(neighborhood, init);
}

}  // namespace bench

#endif  // EXAMPLES_R2PNET_BENCH_METHODS_H_
