// R2P3D_NetFast and R2P3D_HybridFast (see r2p_fast.h).

#include "examples/new_advector/r2p_fast.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "irl/generic_cutting/cut_polygon.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "irl/interface_reconstruction_methods/r2p_neighborhood.h"
#include "irl/interface_reconstruction_methods/reconstruction_interface.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/localizer_link_from_localized_separator_link.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/ml_classifier.h"
#include "examples/new_advector/plicnet.h"
#include "examples/new_advector/r2p_fast_parab.h"
#include "examples/new_advector/r2p_newton_distance.h"
#include "examples/new_advector/r2p_snap.h"
#include "examples/new_advector/r2p_tip_sensor.h"
#include "examples/new_advector/r2p_edge_sensor.h"
#include "examples/new_advector/r2p_edge_topology.h"
#include "examples/new_advector/r2p_nopinch.h"
#include "examples/new_advector/r2pnet.h"
#include "examples/new_advector/reconstruction_types.h"
#include "examples/new_advector/vof_advection.h"

namespace r2pfast {

Profile g_profile;
const char* const kStageNames[12] = {
    "setup (pure cells)", "film phase + classifier", "PLICNet cells",
    "R2P-Net input + MLP", "two-plane Newton",        "one-plane choice",
    "borders",             "pass 2",                  "back-projection",
    "R2P3D (IRL)",         "",                        ""};

}  // namespace r2pfast

#ifdef R2PFAST_PROFILE
#define R2PFAST_TIC(t) const auto t = std::chrono::steady_clock::now()
#define R2PFAST_TOC(t, s)                                                     \
  r2pfast::g_profile.stage[s] +=                                              \
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count()
#define R2PFAST_COUNT(c, n) (r2pfast::g_profile.count[c] += (n))
#else
#define R2PFAST_TIC(t)
#define R2PFAST_TOC(t, s)
#define R2PFAST_COUNT(c, n)
#endif

namespace {

using IRL::global_constants::VF_HIGH;
using IRL::global_constants::VF_LOW;

// ---------------------------------------------------------------------------
// Batched MLPs
// ---------------------------------------------------------------------------

// y[s] = act(b + sum_i x[s][i] W[:,i]) for n samples, with the weights stored
// transposed (row i = input i's weights to every output). Each output sums its
// inputs in increasing i, as the generated get_normal/get_class loops do, so
// results are bit-identical to them (skipped zero inputs add +-0). The weight
// row is reused across the batch, and the output loop vectorises.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
__attribute__((target_clones("avx2", "default")))
#endif
void denseBatch(const double* __restrict wt, const double* __restrict bias, const int in,
                const int out, const double* __restrict x, const int n,
                double* __restrict y, const bool relu) {
  for (int s = 0; s < n; ++s)
    for (int j = 0; j < out; ++j) y[s * out + j] = bias[j];
  for (int i = 0; i < in; ++i) {
    const double* __restrict w = wt + static_cast<long>(i) * out;
    for (int s = 0; s < n; ++s) {
      const double xi = x[static_cast<long>(s) * in + i];
      if (xi == 0.0) continue;
      double* __restrict ys = y + s * out;
      for (int j = 0; j < out; ++j) ys[j] += xi * w[j];
    }
  }
  if (relu)
    for (int q = 0; q < n * out; ++q)
      if (y[q] < 0.0) y[q] = 0.0;
}

// Cells per network call. 1 = every cell is evaluated on its own, as in a
// per-cell code (NGA2); larger values reuse each weight row across the batch
// (same results, less memory traffic).
constexpr int kBatch = 1;
constexpr int kMaxWidth = 512; // widest layer or input

class Mlp {
 public:
  template <int Out, int In>
  void addLayer(const double (&w)[Out][In], const double (&bias)[Out]) {
    Layer l;
    l.in = In;
    l.out = Out;
    l.wt.resize(static_cast<std::size_t>(In) * Out);
    for (int o = 0; o < Out; ++o)
      for (int i = 0; i < In; ++i) l.wt[static_cast<std::size_t>(i) * Out + o] = w[o][i];
    l.b.assign(bias, bias + Out);
    layers_.push_back(std::move(l));
  }
  int in() const { return layers_.front().in; }
  // x: n x in, out: n x (last width). n <= kBatch.
  void forward(const double* x, const int n, double* out) const {
    thread_local std::vector<double> buf[2] = {std::vector<double>(kBatch * kMaxWidth),
                                               std::vector<double>(kBatch * kMaxWidth)};
    const double* src = x;
    for (std::size_t l = 0; l < layers_.size(); ++l) {
      const bool last = l + 1 == layers_.size();
      double* dst = last ? out : buf[l % 2].data();
      denseBatch(layers_[l].wt.data(), layers_[l].b.data(), layers_[l].in, layers_[l].out, src,
                 n, dst, !last);
      src = dst;
    }
  }

 private:
  struct Layer {
    int in = 0, out = 0;
    std::vector<double> wt, b;
  };
  std::vector<Layer> layers_;
};

const Mlp& classifierNet() {
  static const Mlp net = [] {
    namespace d = ml_classifier::detail;
    Mlp m;
    m.addLayer(d::lay1_weight, d::lay1_bias);
    m.addLayer(d::lay2_weight, d::lay2_bias);
    m.addLayer(d::lay3_weight, d::lay3_bias);
    m.addLayer(d::lay4_weight, d::lay4_bias);
    return m;
  }();
  return net;
}

const Mlp& plicNet() {
  static const Mlp net = [] {
    Mlp m;
    m.addLayer(plicnet::lay1_weight, plicnet::lay1_bias);
    m.addLayer(plicnet::lay2_weight, plicnet::lay2_bias);
    m.addLayer(plicnet::lay3_weight, plicnet::lay3_bias);
    m.addLayer(plicnet::lay4_weight, plicnet::lay4_bias);
    return m;
  }();
  return net;
}

const Mlp& r2pNet() {
  static const Mlp net = [] {
    Mlp m;
    m.addLayer(r2pnet::lay1_weight, r2pnet::lay1_bias);
    m.addLayer(r2pnet::lay2_weight, r2pnet::lay2_bias);
    m.addLayer(r2pnet::lay3_weight, r2pnet::lay3_bias);
    m.addLayer(r2pnet::lay4_weight, r2pnet::lay4_bias);
    return m;
  }();
  return net;
}

// ---------------------------------------------------------------------------
// Per-cell inputs
// ---------------------------------------------------------------------------

struct Fields {
  const BasicMesh& mesh;
  const Data<double>& vf;
  const Data<IRL::Pt>& liq;
  const Data<IRL::Pt>& gas;
};

IRL::RectangularCuboid cellCube(const BasicMesh& mesh, const int i, const int j, const int k) {
  return IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                                                 IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
}

// ---------------------------------------------------------------------------
// Per-cell reconstruction timing
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
double seconds(const Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

enum Path { kPlicNet = 0, kR2PNet = 1, kR2P3D = 2 };
const char* const kPathNames[3] = {"PLICNet", "R2P-Net", "R2P3D"};

// One cell's reconstruction time, from after classification to its final
// planes. The hybrid's back-projection (and, with kBatch > 1, a batched
// network call) is shared by several cells; each gets an equal share.
struct CellTime {
  std::array<int, 3> c;
  int path = kPlicNet;
  double input = 0.0, network = 0.0, geometry = 0.0, pass2 = 0.0, backproj = 0.0;
  double opening = -1.0;   // R2P-Net two-plane cells: angle between the predicted faces [deg]
};

// Final film gap: minimum over the cell / value at the centre (two-plane
// cells; 1 = parallel slab, < 0 = the planes cross inside the cell). The gap
// is linear in space, so its minimum is at a corner.
double gapRatio(const BasicMesh& mesh, const IRL::PlanarSeparator& sep, const int i, const int j,
                const int k) {
  if (sep.getNumberOfPlanes() != 2) return std::nan("");
  const double s = sep.isFlipped() ? -1.0 : 1.0;
  auto gap = [&](const IRL::Pt& p) {
    return s * ((sep[0].distance() - sep[0].normal() * p) + (sep[1].distance() - sep[1].normal() * p));
  };
  double gmin = 1e300;
  for (int q = 0; q < 8; ++q)
    gmin = std::min(gmin, gap(IRL::Pt(q & 1 ? mesh.x(i + 1) : mesh.x(i), q & 2 ? mesh.y(j + 1) : mesh.y(j),
                                      q & 4 ? mesh.z(k + 1) : mesh.z(k))));
  return gmin / gap(IRL::Pt(mesh.xm(i), mesh.ym(j), mesh.zm(k)));
}

// With the environment variable R2P_CELL_TIMING set to a file name, every
// call appends one CSV row per mixed cell to that file (truncated when the
// process starts). `call` counts calls of that method, i.e. time steps.
class CellTimingLog {
 public:
  static CellTimingLog& instance() {
    static CellTimingLog log;
    return log;
  }
  bool enabled() const { return file_ != nullptr; }
  void write(const char* method, const long call, const Data<double>& vf,
             const Data<IRL::PlanarSeparator>& iface, const std::vector<CellTime>& rows) {
    for (const CellTime& r : rows) {
      const int i = r.c[0], j = r.c[1], k = r.c[2];
      const double total = r.input + r.network + r.geometry + r.pass2 + r.backproj;
      std::fprintf(file_, "%ld,%s,%s,%d,%d,%d,%.17g,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%.4f\n",
                   call, method, kPathNames[r.path], i, j, k, vf(i, j, k),
                   static_cast<int>(iface(i, j, k).getNumberOfPlanes()), 1e6 * total,
                   1e6 * r.input, 1e6 * r.network, 1e6 * r.geometry, 1e6 * r.pass2,
                   1e6 * r.backproj, r.opening, snapped(i, j, k),
                   gapRatio(vf.getMesh(), iface(i, j, k), i, j, k));
    }
    std::fflush(file_);
  }

 private:
  CellTimingLog() {
    const char* path = std::getenv("R2P_CELL_TIMING");
    if (path == nullptr || *path == '\0') return;
    file_ = std::fopen(path, "w");
    if (file_ == nullptr) {
      std::fprintf(stderr, "R2P_CELL_TIMING: cannot open %s\n", path);
      return;
    }
    std::fprintf(file_, "call,method,path,i,j,k,vf,planes,total_us,input_us,network_us,"
                        "geometry_us,pass2_us,backproj_us,opening_deg,snapped,gap_ratio\n");
  }
  ~CellTimingLog() {
    if (file_ != nullptr) std::fclose(file_);
  }
  std::FILE* file_ = nullptr;
};

// 3^3 moments [vf0, centroid0, centroid1] per cell (centroids relative to
// their cell centre, in cell units), phase 0 = gas when gas_first; m gets
// phase 0's volume and first moments about the centre cell.
void stencilMoments(const Fields& F, const int i, const int j, const int k, const bool gas_first,
                    double* moments, double* m) {
  const BasicMesh& mesh = F.mesh;
  m[0] = m[1] = m[2] = m[3] = 0.0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        const int idx = 7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
        const IRL::Pt& c0 = gas_first ? F.gas(ii, jj, kk) : F.liq(ii, jj, kk);
        const IRL::Pt& c1 = gas_first ? F.liq(ii, jj, kk) : F.gas(ii, jj, kk);
        moments[idx] = gas_first ? 1.0 - F.vf(ii, jj, kk) : F.vf(ii, jj, kk);
        moments[idx + 1] = (c0[0] - mesh.xm(ii)) / mesh.dx();
        moments[idx + 2] = (c0[1] - mesh.ym(jj)) / mesh.dy();
        moments[idx + 3] = (c0[2] - mesh.zm(kk)) / mesh.dz();
        moments[idx + 4] = (c1[0] - mesh.xm(ii)) / mesh.dx();
        moments[idx + 5] = (c1[1] - mesh.ym(jj)) / mesh.dy();
        moments[idx + 6] = (c1[2] - mesh.zm(kk)) / mesh.dz();
        m[0] += moments[idx];
        m[1] += (moments[idx + 1] + (ii - i)) * moments[idx];
        m[2] += (moments[idx + 2] + (jj - j)) * moments[idx];
        m[3] += (moments[idx + 3] + (kk - k)) * moments[idx];
      }
}

// Undo reflect_moments' axis permutation (direction2) then reflection
// (direction) on a network output.
IRL::Normal undoCanonical(IRL::Normal n, const int direction, const int direction2) {
  switch (direction2) {
    case 1: std::swap(n[0], n[1]); break;
    case 2: std::swap(n[1], n[2]); break;
    case 3: std::swap(n[0], n[2]); break;
    case 4: std::swap(n[1], n[2]); std::swap(n[0], n[1]); break;
    case 5: std::swap(n[0], n[2]); std::swap(n[0], n[1]); break;
  }
  switch (direction) {
    case 1: n[0] = -n[0]; break;
    case 2: n[1] = -n[1]; break;
    case 3: n[2] = -n[2]; break;
    case 4: n[0] = -n[0]; n[1] = -n[1]; break;
    case 5: n[0] = -n[0]; n[2] = -n[2]; break;
    case 6: n[1] = -n[1]; n[2] = -n[2]; break;
    case 7: n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; break;
  }
  return n;
}

IRL::Normal toMeshNormal(IRL::Normal n, const BasicMesh& mesh) {
  n[0] *= mesh.dx();
  n[1] *= mesh.dy();
  n[2] *= mesh.dz();
  n.normalize();
  return n;
}

// PLICNet: minority phase (VF >= 0.5: gas) first, canonicalised about its
// centre of mass.
struct PlicInput {
  double moments[189];
  int direction = 0, direction2 = 0;
  bool flip = false;
};

void plicInput(const Fields& F, const int i, const int j, const int k, PlicInput* p) {
  p->flip = F.vf(i, j, k) >= 0.5;
  double m[4];
  stencilMoments(F, i, j, k, p->flip, p->moments, m);
  const double center[3] = {m[1] / m[0], m[2] / m[0], m[3] / m[0]};
  p->direction = p->direction2 = 0;
  plicnet::reflect_moments(p->moments, center, &p->direction, &p->direction2);
}

IRL::Normal plicNormal(const double* out, const PlicInput& p, const BasicMesh& mesh) {
  IRL::Normal n = undoCanonical(IRL::Normal(out[0], out[1], out[2]), p.direction, p.direction2);
  if (!p.flip) n = -n;
  return toMeshNormal(n, mesh);
}

// PLICNet normals and volume-matching planes for a list of cells. `timing`
// (n entries, or nullptr) receives each cell's time.
void plicnetCells(const Fields& F, const std::array<int, 3>* cells, const int n,
                  Data<IRL::PlanarSeparator>* a_interface, CellTime* timing = nullptr) {
  PlicInput in[kBatch];
  double x[kBatch * 189], y[kBatch * 3];
  Clock::time_point t;
  for (int b = 0; b < n; ++b) {
    if (timing) t = Clock::now();
    plicInput(F, cells[b][0], cells[b][1], cells[b][2], &in[b]);
    std::copy(in[b].moments, in[b].moments + 189, x + b * 189);
    if (timing) {
      timing[b] = CellTime();
      timing[b].c = cells[b];
      timing[b].path = kPlicNet;
      timing[b].input = seconds(t);
    }
  }
  if (timing) t = Clock::now();
  plicNet().forward(x, n, y);
  if (timing) {
    const double share = seconds(t) / n;
    for (int b = 0; b < n; ++b) timing[b].network = share;
  }
  for (int b = 0; b < n; ++b) {
    if (timing) t = Clock::now();
    const int i = cells[b][0], j = cells[b][1], k = cells[b][2];
    const IRL::Normal normal = plicNormal(y + 3 * b, in[b], F.mesh);
    const double distance = IRL::findDistanceOnePlane(cellCube(F.mesh, i, j, k), F.vf(i, j, k), normal);
    (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal, distance));
    if (timing) timing[b].geometry = seconds(t);
  }
}

// Sphericity (smallest / largest covariance eigenvalue) of one phase's
// centroids over the 3^3 cells holding it; -1 with fewer than 3 points.
double phaseSphericity(const Fields& F, const bool gas, const int i, const int j, const int k) {
  std::array<Eigen::Vector3d, 27> pts;
  int n = 0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        const double vf = F.vf(ii, jj, kk);
        if ((gas ? 1.0 - vf : vf) <= VF_LOW) continue;
        const IRL::Pt& p = gas ? F.gas(ii, jj, kk) : F.liq(ii, jj, kk);
        pts[n++] = Eigen::Vector3d(p[0], p[1], p[2]);
      }
  if (n < 3) return -1.0;
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (int q = 0; q < n; ++q) c += pts[q];
  c /= static_cast<double>(n);
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (int q = 0; q < n; ++q) cov += (pts[q] - c) * (pts[q] - c).transpose();
  const Eigen::Vector3d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(cov).eigenvalues();
  return ev(2) > 1.0e-30 ? std::max(0.0, ev(0)) / ev(2) : -1.0;
}

double stencilVfSum(const Fields& F, const int i, const int j, const int k) {
  double vol = 0.0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) vol += F.vf(ii, jj, kk);
  return vol;
}

// Phase 0 (true = gas): the flatter centroid cloud, i.e. the film.
bool filmIsGas(const Fields& F, const int i, const int j, const int k) {
  const double s_liq = phaseSphericity(F, false, i, j, k);
  const double s_gas = phaseSphericity(F, true, i, j, k);
  if (s_liq < 0.0 || s_gas < 0.0) return stencilVfSum(F, i, j, k) >= 0.5 * 27.0;
  return s_gas < s_liq;
}

// Classifier input for cell (i,j,k) with phase 0 = gas when `gas`. False when
// get_class would return 0 without evaluating the network.
bool classifierInput(const Fields& F, const int i, const int j, const int k, const bool gas,
                     double* flat) {
  const BasicMesh& mesh = F.mesh;
  ml_classifier::Stencil st;
  const IRL::Pt cell_center(mesh.xm(i), mesh.ym(j), mesh.zm(k));
  for (int ii = 0; ii < 5; ++ii)
    for (int jj = 0; jj < 5; ++jj)
      for (int kk = 0; kk < 5; ++kk) {
        const int gi = i + ii - 2, gj = j + jj - 2, gk = k + kk - 2;
        double vf = F.vf(gi, gj, gk);
        if (gas) vf = 1 - vf;
        st.f(ii, jj, kk) = vf;
        IRL::Pt bary = gas ? F.gas(gi, gj, gk) : F.liq(gi, gj, gk);
        bary -= cell_center;
        bary[0] /= mesh.dx();
        bary[1] /= mesh.dy();
        bary[2] /= mesh.dz();
        bary *= vf;
        st.b(ii, jj, kk, 0) = bary[0];
        st.b(ii, jj, kk, 1) = bary[1];
        st.b(ii, jj, kk, 2) = bary[2];
      }
  if (st.f(ml_classifier::CID, ml_classifier::CID, ml_classifier::CID) <
      ml_classifier::EPSILON_CONNECT)
    return false;
  double tmp[ml_classifier::NIN];
  ml_classifier::detail::preprocess_and_flatten(st, tmp);
  std::copy(tmp, tmp + ml_classifier::NIN, flat);
  return true;
}

// Classes (get_class ids) for a batch of cells, and whether each goes to R2P.
void classify(const Fields& F, const std::array<int, 3>* cells, const bool* gas, const int n,
              int* cls, bool* use_r2p) {
  const BasicMesh& mesh = F.mesh;
  static thread_local std::vector<double> flat(kBatch * ml_classifier::NIN);
  double logits[kBatch * 6];
  int slot[kBatch];
  int nin = 0;
  for (int b = 0; b < n; ++b) {
    const int i = cells[b][0], j = cells[b][1], k = cells[b][2];
    use_r2p[b] = false;
    cls[b] = 0;
    const bool stencil_available =
        i - 2 >= mesh.imino() && i + 2 <= mesh.imaxo() && j - 2 >= mesh.jmino() &&
        j + 2 <= mesh.jmaxo() && k - 2 >= mesh.kmino() && k + 2 <= mesh.kmaxo();
    if (stencil_available &&
        classifierInput(F, i, j, k, gas[b], flat.data() + nin * ml_classifier::NIN))
      slot[nin++] = b;
  }
  classifierNet().forward(flat.data(), nin, logits);
  for (int q = 0; q < nin; ++q) {
    const double* l = logits + 6 * q;
    int best = 0;
    for (int c = 1; c < 6; ++c)
      if (l[c] > l[best]) best = c;
    cls[slot[q]] = best + 1;
    use_r2p[slot[q]] = (best + 1 == 4 || best + 1 == 6);   // sheet, sheet end
  }
  // Thin-film guard: the other phase on both sides of the film, apart, means
  // the film continues here and needs two planes (r2p_edge_topology.h)
  for (int b = 0; b < n; ++b) {
    const int i = cells[b][0], j = cells[b][1], k = cells[b][2];
    if (r2pedgetopo::filmSeparates(F.vf, gas[b], i, j, k)) {
      film_guard(i, j, k) = use_r2p[b] ? 2 : 1;
      use_r2p[b] = true;
    }
    // Very thin film: R2P-Net, which makes it a PCA slab (r2p_snap.h veryThinStencil)
    if (!use_r2p[b] && r2psnap::veryThinStencil(mesh, F.vf, F.liq, F.gas, i, j, k, gas[b])) use_r2p[b] = true;
    if (!use_r2p[b]) {
      const bool stencil_available =
          i - 2 >= mesh.imino() && i + 2 <= mesh.imaxo() && j - 2 >= mesh.jmino() &&
          j + 2 <= mesh.jmaxo() && k - 2 >= mesh.kmino() && k + 2 <= mesh.kmaxo();
      one_plane_reason(i, j, k) = stencil_available ? 1 : 2;
    }
  }
}

// R2P-Net input: film phase first, canonicalised about the PCA direction of
// the film centroids (oriented toward the film's centre of mass).
struct R2PInput {
  double input[192];
  int direction = 0, direction2 = 0;
  IRL::Normal pca;   // PCA direction of the 3^3 film-phase centroids (physical)
  int npts = 0;      // number of those centroids
};

IRL::Normal pcaNormal(const IRL::Pt* points, const int N) {
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (int q = 0; q < N; ++q) centroid += Eigen::Vector3d(points[q][0], points[q][1], points[q][2]);
  centroid = centroid / double(N);
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (int q = 0; q < N; ++q) {
    const Eigen::Vector3d d = Eigen::Vector3d(points[q][0], points[q][1], points[q][2]) - centroid;
    covariance += d * d.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
  const Eigen::Vector3d local_z = eigensolver.eigenvectors().col(0).normalized();
  IRL::Normal n(local_z[0], local_z[1], local_z[2]);
  n.normalize();
  return n;
}

void r2pInput(const Fields& F, const int i, const int j, const int k, const bool gas, R2PInput* r) {
  double moments[189], m[4];
  stencilMoments(F, i, j, k, gas, moments, m);
  std::array<IRL::Pt, 27> points;
  int np = 0;
  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        const double vf = F.vf(ii, jj, kk);
        if (!gas && vf > VF_LOW) points[np++] = F.liq(ii, jj, kk);
        if (gas && (1 - vf) > VF_LOW) points[np++] = F.gas(ii, jj, kk);
      }
  IRL::Normal dir = pcaNormal(points.data(), np);
  if (IRL::dotProduct(dir, IRL::Pt(m[1] / m[0], m[2] / m[0], m[3] / m[0])) < 0) dir = -dir;
  r->pca = dir;
  r->npts = np;

  double center[3] = {dir[0], dir[1], dir[2]};
  r->direction = r->direction2 = 0;
  r2pnet::reflect_moments(moments, center, &r->direction, &r->direction2);
  switch (r->direction) {
    case 1: center[0] = -center[0]; break;
    case 2: center[1] = -center[1]; break;
    case 3: center[2] = -center[2]; break;
    case 4: center[0] = -center[0]; center[1] = -center[1]; break;
    case 5: center[0] = -center[0]; center[2] = -center[2]; break;
    case 6: center[1] = -center[1]; center[2] = -center[2]; break;
    case 7: center[0] = -center[0]; center[1] = -center[1]; center[2] = -center[2]; break;
  }
  switch (r->direction2) {
    case 1: std::swap(center[0], center[1]); break;
    case 2: std::swap(center[1], center[2]); break;
    case 3: std::swap(center[0], center[2]); break;
    case 4: std::swap(center[0], center[1]); std::swap(center[1], center[2]); break;
    case 5: std::swap(center[0], center[1]); std::swap(center[0], center[2]); break;
  }
  std::copy(moments, moments + 189, r->input);
  r->input[189] = center[0];
  r->input[190] = center[1];
  r->input[191] = center[2];
}

// One plane from `nrm` at the cell's volume fraction, scored by the distance
// of its liquid and gas centroids to the cell's; DBL_MAX if unusable.
double onePlaneScore(const Fields& F, const int i, const int j, const int k,
                     const IRL::RectangularCuboid& cube, IRL::Normal nrm, IRL::PlanarSeparator* out) {
  if (nrm.calculateMagnitude() < 0.5) return DBL_MAX;
  nrm.normalize();
  const double vf = F.vf(i, j, k);
  *out = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(nrm, IRL::findDistanceOnePlane(cube, vf, nrm)));
  const auto svm = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(cube, *out);
  if (std::abs(svm[0].volume() / cube.calculateVolume() - vf) > 1.0e-6) return DBL_MAX;
  double err = 0.0;
  if (vf > VF_LOW) err += IRL::magnitude(F.liq(i, j, k) - svm[0].centroid());
  if (vf < VF_HIGH) err += IRL::magnitude(F.gas(i, j, k) - svm[1].centroid());
  return err;
}

// Mixed interior cells, in (i,j,k) order.
std::vector<std::array<int, 3>> listMixed(const Fields& F) {
  const BasicMesh& mesh = F.mesh;
  std::vector<std::array<int, 3>> mixed;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i)
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j)
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        const double vf = F.vf(i, j, k);
        if (!(vf < VF_LOW || vf > VF_HIGH)) mixed.push_back({i, j, k});
      }
  return mixed;
}

// Pure interior cells get a plane-free separator of the right phase.
void setPure(const Fields& F, Data<IRL::PlanarSeparator>* a_interface) {
  const BasicMesh& mesh = F.mesh;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i)
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j)
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        const double vf = F.vf(i, j, k);
        if (vf < VF_LOW || vf > VF_HIGH) {
          (*a_interface)(i, j, k) = IRL::PlanarSeparator::fromOnePlane(
              IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), std::copysign(1.0, vf - 0.5)));
          recon_method(i, j, k) = -1;
          num_planes(i, j, k) = feature_class(i, j, k) = branch(i, j, k) = snapped(i, j, k) = 0;
        }
      }
}

// Diagnostics read by writeOutInterface (reconstruction_types.h globals):
// recon_method 0 = PLICNet, 1 = R2P; feature_class = classifier id; branch
// 0 = PLICNet, 1 = one plane from PLICNet, 2 = two planes, 3 = one plane from
// R2P-Net, 6 = refined by pass 2. num_planes is filled in by finishDiagnostics.
void setDiagnostics(const std::array<int, 3>& c, const int method, const int cls, const int br) {
  recon_method(c[0], c[1], c[2]) = method;
  feature_class(c[0], c[1], c[2]) = cls;
  branch(c[0], c[1], c[2]) = br;
}

void finishDiagnostics(const std::vector<std::array<int, 3>>& mixed,
                       const Data<IRL::PlanarSeparator>& a_interface) {
  for (const auto& c : mixed)
    num_planes(c[0], c[1], c[2]) = static_cast<int>(a_interface(c[0], c[1], c[2]).getNumberOfPlanes());
}

}  // namespace

// ===========================================================================
void R2P3D_NetFast::getReconstruction(
    const Data<double>& a_liquid_volume_fraction, const Data<IRL::Pt>& a_liquid_centroid,
    const Data<IRL::Pt>& a_gas_centroid, const Data<IRL::LocalizedSeparatorLink>&,
    const double, const Data<double>&, const Data<double>&, const Data<double>&,
    Data<IRL::PlanarSeparator>* a_interface) {
  const Fields F{a_liquid_volume_fraction.getMesh(), a_liquid_volume_fraction, a_liquid_centroid,
                 a_gas_centroid};
  const BasicMesh& mesh = F.mesh;

  R2PFAST_TIC(t_setup);
  const std::vector<std::array<int, 3>> mixed = listMixed(F);
  setPure(F, a_interface);
  R2PFAST_TOC(t_setup, 0);
  R2PFAST_COUNT(0, static_cast<long>(mixed.size()));

  CellTimingLog& log = CellTimingLog::instance();
  const bool timed = log.enabled();
  static long call = 0;
  ++call;
  plicNet();   // build the weight tables outside the timed regions
  r2pNet();
  std::vector<CellTime> rows;
  std::vector<std::size_t> two_plane_row;   // rows index of each two_plane cell

  std::vector<std::array<int, 3>> two_plane;
  for (std::size_t c0 = 0; c0 < mixed.size(); c0 += kBatch) {
    const int n = static_cast<int>(std::min<std::size_t>(kBatch, mixed.size() - c0));
    const std::array<int, 3>* cells = mixed.data() + c0;

    // Film phase and routing.
    R2PFAST_TIC(t0);
    bool gas[kBatch], use_r2p[kBatch];
    int cls[kBatch];
    for (int b = 0; b < n; ++b) {
      gas[b] = filmIsGas(F, cells[b][0], cells[b][1], cells[b][2]);
      tip_sensor(cells[b][0], cells[b][1], cells[b][2]) =
          r2ptip::isTip(F.vf, F.liq, F.gas, cells[b][0], cells[b][1], cells[b][2], gas[b]);
    }
    classify(F, cells, gas, n, cls, use_r2p);
    for (int b = 0; b < n; ++b) setDiagnostics(cells[b], use_r2p[b] ? 1 : 0, cls[b], 0);
    R2PFAST_TOC(t0, 1);

    std::array<int, 3> plic[kBatch];
    int np = 0, nr = 0, r2p_slot[kBatch];
    for (int b = 0; b < n; ++b) {
      if (use_r2p[b]) r2p_slot[nr++] = b;
      else plic[np++] = cells[b];
    }
    R2PFAST_COUNT(1, nr);

    // Everything else: PLICNet.
    R2PFAST_TIC(t2);
    if (np > 0) {
      CellTime pt[kBatch];
      plicnetCells(F, plic, np, a_interface, timed ? pt : nullptr);
      if (timed) rows.insert(rows.end(), pt, pt + np);
    }
    R2PFAST_TOC(t2, 2);
    if (nr == 0) continue;

    // Sheets and sheet ends: R2P-Net.
    R2PFAST_TIC(t3);
    R2PInput in[kBatch];
    CellTime rt[kBatch];
    double x[kBatch * 192], y[kBatch * 6];
    Clock::time_point t;
    for (int q = 0; q < nr; ++q) {
      if (timed) t = Clock::now();
      const std::array<int, 3>& c = cells[r2p_slot[q]];
      r2pInput(F, c[0], c[1], c[2], gas[r2p_slot[q]], &in[q]);
      std::copy(in[q].input, in[q].input + 192, x + 192 * q);
      if (timed) {
        rt[q].c = c;
        rt[q].path = kR2PNet;
        rt[q].input = seconds(t);
      }
    }
    if (timed) t = Clock::now();
    r2pNet().forward(x, nr, y);
    if (timed) {
      const double share = seconds(t) / nr;
      for (int q = 0; q < nr; ++q) rt[q].network = share;
    }
    R2PFAST_TOC(t3, 3);

    for (int q = 0; q < nr; ++q) {
      if (timed) t = Clock::now();
      const int b = r2p_slot[q];
      const int i = cells[b][0], j = cells[b][1], k = cells[b][2];
      const double vf = F.vf(i, j, k);
      const IRL::RectangularCuboid cube = cellCube(mesh, i, j, k);
      IRL::Normal n1 = undoCanonical(IRL::Normal(y[6 * q], y[6 * q + 1], y[6 * q + 2]),
                                     in[q].direction, in[q].direction2);
      IRL::Normal n2 = undoCanonical(IRL::Normal(y[6 * q + 3], y[6 * q + 4], y[6 * q + 5]),
                                     in[q].direction, in[q].direction2);
      // Very thin film: a slab along the PCA direction instead of the
      // network's normals (r2p_snap.h pcaSlab)
      if (in[q].npts >= 6 && r2psnap::pcaSlab(mesh, F.vf, F.liq, F.gas, i, j, k, gas[b], in[q].pca, n1, n2))
        snapped(i, j, k) = 3;
      edge_sensor(i, j, k) = r2pedge::edgeCount(F.vf, gas[b], i, j, k, r2pedge::meanNormal(n1, n2, mesh));
      edge_topo(i, j, k) = r2pedgetopo::gasWraps(F.vf, gas[b], i, j, k, r2pedge::meanNormal(n1, n2, mesh));
      bool one_plane = n2.calculateMagnitude() < 0.85 || n1.calculateMagnitude() < 0.85;
      // Where the thin-film guard holds the film runs through the cell: a
      // parallel slab instead of one plane (r2p_snap.h guardSlab)
      if (one_plane && film_guard(i, j, k) != 0 && r2psnap::guardSlab(n1, n2)) {
        one_plane = false;
        snapped(i, j, k) = 2;
      }
      IRL::PlanarSeparator& sep = (*a_interface)(i, j, k);

      if (!one_plane) {
        // Two planes around the film, distances from VF and the film centroid.
        R2PFAST_TIC(t4);
        n1 = toMeshNormal(n1, mesh);
        n2 = toMeshNormal(n2, mesh);
        // Thin film with a noise-level opening: parallel planes (r2p_snap.h)
        if (timed) rt[q].opening = std::acos(std::max(-1.0, std::min(1.0, -IRL::dotProduct(n1, n2)))) * 180.0 / M_PI;
        if (r2psnap::snapThinFilm(cube, gas[b] ? 1.0 - vf : vf, gas[b] ? F.gas(i, j, k) : F.liq(i, j, k),
                                  cls[b], n1, n2))
          snapped(i, j, k) = 1;
        if (!gas[b]) {
          n1 = -n1;
          n2 = -n2;
        }
        sep = IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(n1, 0), IRL::Plane(n2, 0),
                                                  gas[b] ? -1 : 1);
        r2pnewton::R2PNewtonDistanceSolver(vf, F.liq(i, j, k), F.gas(i, j, k), sep, cube);
        if (sep.getNumberOfPlanes() != 2) one_plane_reason(i, j, k) = 4;
        // Unless the film ends here, the planes may not pinch it off (r2p_nopinch.h)
        r2pnopinch::applyAt(mesh, i, j, k, vf, F.liq(i, j, k), F.gas(i, j, k), &sep);
        if (one_plane_reason(i, j, k) == 0 && sep.getNumberOfPlanes() != 2) one_plane_reason(i, j, k) = 5;
        if (sep.getNumberOfPlanes() == 2) {
          two_plane.push_back(cells[b]);
          two_plane_row.push_back(rows.size());
        }
        branch(i, j, k) = 2;
        R2PFAST_TOC(t4, 4);
        if (timed) {
          rt[q].geometry = seconds(t);
          rows.push_back(rt[q]);
        }
        continue;
      }

      // One plane: the network's surviving normal or PLICNet's, whichever
      // better matches the cell's centroids.
      R2PFAST_TIC(t5);
      one_plane_reason(i, j, k) = 3;
      IRL::Normal nn = n2.calculateMagnitude() < n1.calculateMagnitude() ? n1 : n2;
      nn[0] *= mesh.dx();
      nn[1] *= mesh.dy();
      nn[2] *= mesh.dz();
      if (nn.calculateMagnitude() > 0.0) nn.normalize();
      if (!gas[b]) nn = -nn;
      if (IRL::dotProduct(nn, F.liq(i, j, k) - cube.calculateCentroid()) > 0) nn = -nn;
      IRL::PlanarSeparator sep_nn, sep_plic;
      const double err_nn = onePlaneScore(F, i, j, k, cube, nn, &sep_nn);
      PlicInput p;
      plicInput(F, i, j, k, &p);
      double out[3];
      Clock::time_point tn;
      if (timed) tn = Clock::now();
      plicNet().forward(p.moments, 1, out);
      const double plic_seconds = timed ? seconds(tn) : 0.0;
      const double err_plic = onePlaneScore(F, i, j, k, cube, plicNormal(out, p, mesh), &sep_plic);
      sep = err_plic < err_nn ? sep_plic : sep_nn;
      setDiagnostics(cells[b], err_plic < err_nn ? 0 : 1, cls[b], err_plic < err_nn ? 1 : 3);
      R2PFAST_TOC(t5, 5);
      if (timed) {
        rt[q].network += plic_seconds;
        rt[q].geometry = seconds(t) - plic_seconds;
        rows.push_back(rt[q]);
      }
    }
  }
  R2PFAST_COUNT(2, static_cast<long>(two_plane.size()));

  R2PFAST_TIC(t6);
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
  R2PFAST_TOC(t6, 6);

  // Pass 2: paraboloid refinement of the two-plane cells.
  R2PFAST_TIC(t7);
  std::vector<double> pass2_seconds;
  for (const auto& c : r2pfastparab::run(a_liquid_volume_fraction, a_liquid_centroid,
                                         a_gas_centroid, two_plane, a_interface,
                                         timed ? &pass2_seconds : nullptr))
    branch(c[0], c[1], c[2]) = 6;
  correctInterfacePlaneBorders(a_interface);
  finishDiagnostics(mixed, *a_interface);
  R2PFAST_TOC(t7, 7);

  if (timed) {
    for (std::size_t q = 0; q < two_plane_row.size(); ++q) rows[two_plane_row[q]].pass2 = pass2_seconds[q];
    log.write("R2P3D_NetFast", call, a_liquid_volume_fraction, *a_interface, rows);
  }
}

// ===========================================================================
void R2P3D_HybridFast::getReconstruction(
    const Data<double>& a_liquid_volume_fraction, const Data<IRL::Pt>& a_liquid_centroid,
    const Data<IRL::Pt>& a_gas_centroid,
    const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link, const double a_dt,
    const Data<double>& a_U, const Data<double>& a_V, const Data<double>& a_W,
    Data<IRL::PlanarSeparator>* a_interface) {
  const Fields F{a_liquid_volume_fraction.getMesh(), a_liquid_volume_fraction, a_liquid_centroid,
                 a_gas_centroid};
  const BasicMesh& mesh = F.mesh;

  R2PFAST_TIC(t_setup);
  const std::vector<std::array<int, 3>> mixed = listMixed(F);
  R2PFAST_TOC(t_setup, 0);
  R2PFAST_COUNT(0, static_cast<long>(mixed.size()));

  // Route every mixed cell (phase 0 = minority phase of the 3^3 block). The
  // old interface is still needed by the back-projection, so nothing is
  // written until after it.
  std::vector<std::array<int, 3>> r2p_cells, plic_cells;
  for (std::size_t c0 = 0; c0 < mixed.size(); c0 += kBatch) {
    const int n = static_cast<int>(std::min<std::size_t>(kBatch, mixed.size() - c0));
    const std::array<int, 3>* cells = mixed.data() + c0;
    R2PFAST_TIC(t0);
    bool gas[kBatch], use_r2p[kBatch];
    int cls[kBatch];
    for (int b = 0; b < n; ++b)
      gas[b] = stencilVfSum(F, cells[b][0], cells[b][1], cells[b][2]) >= 0.5 * 27.0;
    classify(F, cells, gas, n, cls, use_r2p);
    for (int b = 0; b < n; ++b) {
      (use_r2p[b] ? r2p_cells : plic_cells).push_back(cells[b]);
      setDiagnostics(cells[b], use_r2p[b] ? 1 : 0, cls[b], 0);
    }
    R2PFAST_TOC(t0, 1);
  }
  R2PFAST_COUNT(1, static_cast<long>(r2p_cells.size()));

  CellTimingLog& log = CellTimingLog::instance();
  const bool timed = log.enabled();
  static long call = 0;
  ++call;
  plicNet();   // build the weight tables outside the timed regions
  r2pNet();
  std::vector<CellTime> rows;
  double backproj_seconds = 0.0;
  Clock::time_point t;

  std::vector<IRL::ListedVolumeMoments<IRL::VolumeMomentsAndNormal>> listed(r2p_cells.size());
  if (!r2p_cells.empty()) {
    if (timed) t = Clock::now();
    // Advected normals for the R2P cells: back-project the old interface from
    // only the source cells whose polygons can land in one. RK4 moves a vertex
    // at most max|u| dt per direction.
    R2PFAST_TIC(t8);
    const int nxo = mesh.getNxo(), nyo = mesh.getNyo(), nzo = mesh.getNzo();
    auto lin = [&](const int i, const int j, const int k) {
      return (static_cast<std::size_t>(i - mesh.imino()) * nyo + (j - mesh.jmino())) * nzo +
             (k - mesh.kmino());
    };
    double umax[3] = {0.0, 0.0, 0.0};
    for (int i = mesh.imino(); i <= mesh.imaxo(); ++i)
      for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j)
        for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) {
          umax[0] = std::max(umax[0], std::abs(a_U(i, j, k)));
          umax[1] = std::max(umax[1], std::abs(a_V(i, j, k)));
          umax[2] = std::max(umax[2], std::abs(a_W(i, j, k)));
        }
    const int reach[3] = {static_cast<int>(std::ceil(umax[0] * std::abs(a_dt) / mesh.dx())) + 1,
                          static_cast<int>(std::ceil(umax[1] * std::abs(a_dt) / mesh.dy())) + 1,
                          static_cast<int>(std::ceil(umax[2] * std::abs(a_dt) / mesh.dz())) + 1};
    std::vector<int> target(static_cast<std::size_t>(nxo) * nyo * nzo, -1);
    std::vector<char> source(target.size(), 0);
    for (std::size_t t = 0; t < r2p_cells.size(); ++t) {
      const int i = r2p_cells[t][0], j = r2p_cells[t][1], k = r2p_cells[t][2];
      target[lin(i, j, k)] = static_cast<int>(t);
      for (int ii = std::max(i - reach[0], mesh.imino() + 1); ii <= std::min(i + reach[0], mesh.imaxo() - 1); ++ii)
        for (int jj = std::max(j - reach[1], mesh.jmino() + 1); jj <= std::min(j + reach[1], mesh.jmaxo() - 1); ++jj)
          for (int kk = std::max(k - reach[2], mesh.kmino() + 1); kk <= std::min(k + reach[2], mesh.kmaxo() - 1); ++kk)
            source[lin(ii, jj, kk)] = 1;
    }
    for (int i = mesh.imino() + 1; i <= mesh.imaxo() - 1; ++i)
      for (int j = mesh.jmino() + 1; j <= mesh.jmaxo() - 1; ++j)
        for (int k = mesh.kmino() + 1; k <= mesh.kmaxo() - 1; ++k) {
          if (!source[lin(i, j, k)]) continue;
          R2PFAST_COUNT(3, 1);
          const IRL::PlanarSeparator& sep = (*a_interface)(i, j, k);
          const auto cell = cellCube(mesh, i, j, k);
          const auto localizer_link =
              IRL::LocalizerLinkFromLocalizedSeparatorLink(&a_localized_separator_link(i, j, k));
          for (IRL::UnsignedIndex_t n = 0; n < sep.getNumberOfPlanes(); ++n) {
            IRL::Polygon poly =
                IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, sep, sep[n]);
            if (poly.getNumberOfVertices() == 0) continue;
            for (IRL::UnsignedIndex_t tri = 0; tri < poly.getNumberOfSimplicesInDecomposition(); ++tri) {
              IRL::Tri simplex = static_cast<IRL::Tri>(poly.getSimplexFromDecomposition(tri));
              for (auto& vertex : simplex) vertex = back_project_vertex(vertex, a_dt, a_U, a_V, a_W);
              simplex.calculateAndSetPlaneOfExistence();
              const auto moments = IRL::getVolumeMoments<
                  IRL::TaggedAccumulatedListedVolumeMoments<IRL::VolumeMomentsAndNormal>>(simplex, localizer_link);
              for (IRL::UnsignedIndex_t m = 0; m < moments.size(); ++m) {
                const auto idx = getIndexFromTag(mesh, moments.getTagForIndex(m));
                const int t = target[lin(idx[0], idx[1], idx[2])];
                if (t >= 0) listed[t] += moments.getMomentsForIndex(m);
              }
            }
          }
        }
    for (auto& list : listed)
      for (IRL::UnsignedIndex_t n = list.size() - 1; n != static_cast<IRL::UnsignedIndex_t>(-1); --n) {
        IRL::VolumeMomentsAndNormal& moment = list[n];
        moment.normalizeByVolume();
        moment.normal().normalize();
        if (moment.normal().calculateMagnitude() < 0.95) list.erase(n);
        else moment.multiplyByVolume();
      }
    R2PFAST_TOC(t8, 8);
    if (timed) backproj_seconds = seconds(t);
  }

  R2PFAST_TIC(t_pure);
  setPure(F, a_interface);
  R2PFAST_TOC(t_pure, 0);
  R2PFAST_TIC(t2);
  for (std::size_t c0 = 0; c0 < plic_cells.size(); c0 += kBatch) {
    const int n = static_cast<int>(std::min<std::size_t>(kBatch, plic_cells.size() - c0));
    CellTime pt[kBatch];
    plicnetCells(F, plic_cells.data() + c0, n, a_interface, timed ? pt : nullptr);
    if (timed) rows.insert(rows.end(), pt, pt + n);
  }
  R2PFAST_TOC(t2, 2);

  if (!r2p_cells.empty()) {
    // IRL R2P3D, started from the advected normals (MOF without them).
    R2PFAST_TIC(t9);
    IRL::R2PNeighborhood<IRL::RectangularCuboid> neighborhood;
    neighborhood.resize(27);
    neighborhood.setCenterOfStencil(13);
    IRL::RectangularCuboid stencil_cells[27];
    IRL::SeparatedMoments<IRL::VolumeMoments> stencil_moments[27];
    for (std::size_t t = 0; t < r2p_cells.size(); ++t) {
      Clock::time_point tc;
      if (timed) tc = Clock::now();
      const int i = r2p_cells[t][0], j = r2p_cells[t][1], k = r2p_cells[t][2];
      for (int ii = i - 1; ii < i + 2; ++ii)
        for (int jj = j - 1; jj < j + 2; ++jj)
          for (int kk = k - 1; kk < k + 2; ++kk) {
            const int ind = (ii - i + 1) * 9 + (jj - j + 1) * 3 + (kk - k + 1);
            stencil_cells[ind] = cellCube(mesh, ii, jj, kk);
            const double vol = stencil_cells[ind].calculateVolume();
            stencil_moments[ind] = IRL::SeparatedMoments<IRL::VolumeMoments>(
                IRL::VolumeMoments(F.vf(ii, jj, kk) * vol, F.liq(ii, jj, kk)),
                IRL::VolumeMoments((1.0 - F.vf(ii, jj, kk)) * vol, F.gas(ii, jj, kk)));
            neighborhood.setMember(static_cast<IRL::UnsignedIndex_t>(ind), &stencil_cells[ind],
                                   &stencil_moments[ind]);
          }
      IRL::PlanarSeparator& sep = (*a_interface)(i, j, k);
      if (listed[t].size() == 0) {
        const auto& cell = stencil_cells[13];
        sep = IRL::reconstructionWithMOF3D(cell, stencil_moments[13]);
        neighborhood.setSurfaceArea(getReconstructionSurfaceArea(cell, sep));
      } else {
        sep = IRL::reconstructionWithAdvectedNormals(listed[t], neighborhood);
        double area_sum = 0.0;
        for (const auto& moment : listed[t]) area_sum += moment.volumeMoments().volume();
        neighborhood.setSurfaceArea(area_sum);
      }
      sep = reconstructionWithR2P3D(neighborhood, sep);
      if (timed) {
        CellTime r;
        r.c = r2p_cells[t];
        r.path = kR2P3D;
        r.geometry = seconds(tc);
        r.backproj = backproj_seconds / static_cast<double>(r2p_cells.size());
        rows.push_back(r);
      }
    }
    R2PFAST_TOC(t9, 9);
  }

  R2PFAST_TIC(t6);
  a_interface->updateBorder();
  correctInterfacePlaneBorders(a_interface);
  R2PFAST_TOC(t6, 6);
  finishDiagnostics(mixed, *a_interface);
  if (timed) log.write("R2P3D_HybridFast", call, a_liquid_volume_fraction, *a_interface, rows);
}
