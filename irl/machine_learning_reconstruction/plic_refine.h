#ifndef PLIC_REFINE_H_
#define PLIC_REFINE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/moments/separated_volume_moments.h"
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "irl/machine_learning_reconstruction/plic_geometry.h"
#include "irl/machine_learning_reconstruction/plic_paraboloid.h"

// One-to-one C++ equivalent of the Fortran converge_plicnet.
//
// Gate (is_locally_flat): each neighbor carries its OWN already-reconstructed
// plane. Two tests, verbatim from the Fortran: neighbor normals must agree
// with ours to within flat_mindot, and neighbor interface polygon centroids
// must lie within flat_maxdev cell widths of our own plane. Flatness is
// derived from the worst normal dot product and scales the regularization.
//
// Fit (PlaneFit + runLM): 2-parameter tilt off the network normal, residual
// of 1 VF row + 3 liquid + 3 gas centroid rows per face-stencil cell, plus 2
// flatness-scaled regularization rows. Distance is never a free parameter --
// IRL re-solves it against the center cell's VF at every trial.
namespace plicfit {

struct Options {
  // Orientation source:
  //   0 = Levenberg-Marquardt plane fit against cut moments (converge_plicnet)
  //   1 = pointwise local paraboloid fit (paraboloid_fit)
  //   2 = integral local paraboloid fit (paraboloid_integral_fit / Jibben)
  int orientation_method = 2;
  bool parab_use_gate = true;

  int parab_minpts = 6;          // 6 unknowns in the quadratic form
  double parab_h = 2.5;          // wgauss support radius, in cell widths
  double parab_maxresid = 0.25;  // reject if RMS LS residual exceeds this

  double regularization = 1.0;
  double max_rotation = 0.35;
  double fd_step = 1.0e-6;
  int max_ite = 30;
  int max_trial = 12;
  double flat_mindot = -100;   // ~10 deg
  //double flat_maxdev = 0.03;    // cell widths
  double flat_maxdev = 0.15; 
  int flat_minnbr = 3;
  double volume_tolerance = 1.0e-14;
};

struct Result {
  bool fitted = false;
  double flatness = 1.0;
  double residual = 0.0;
  int iterations = 0;
};

// One cell of the FIT stencil: geometry plus its true target moments.
struct StencilCell {
  IRL::RectangularCuboid cell;
  double volume = 1.0;
  double vf = 0.0;
  IRL::Pt liquid_bary;
  IRL::Pt gas_bary;
  bool liquid_meaningful = false;
  bool gas_meaningful = false;
};

// One cell's OWN reconstructed plane -- the offline equivalent of an entry
// in interface_polygon(:,i,j,k). Index 0 must be the center cell.
struct NeighborPlane {
  IRL::Pt lo, hi;
  bool mixed = false;
  IRL::Normal normal;
  double distance = 0.0;
};

namespace detail {

inline IRL::Normal perpendicularTo(const IRL::Normal& v) {
  IRL::Normal p = IRL::crossProduct(
      v, std::abs(v[0]) < 0.9 ? IRL::Normal(1, 0, 0) : IRL::Normal(0, 1, 0));
  p.normalize();
  return p;
}

inline IRL::Normal rotate(const IRL::Normal& n0, const IRL::Normal& axis,
                          const double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  IRL::Normal out =
      n0 * c + IRL::crossProduct(axis, n0) * s + axis * (axis * n0) * (1.0 - c);
  out.normalize();
  return out;
}

}  // namespace detail

class PlaneFit {
 public:
  PlaneFit(const std::vector<StencilCell>& stencil, const IRL::Normal& guess,
           const Options& options)
      : stencil_(stencil), options_(options) {
    n0_ = guess;
    n0_.normalize();
    t0_ = detail::perpendicularTo(n0_);
    t1_ = IRL::crossProduct(n0_, t0_);
    t1_.normalize();
    const auto& c = stencil_[0].cell;
    width_ = (c.calculateSideLength(0) + c.calculateSideLength(1) +
              c.calculateSideLength(2)) / 3.0;
    pivot_ = c.calculateCentroid();
    n_fit_rows_ = 7 * static_cast<int>(stencil_.size());
    reg_base_ =
        std::sqrt(options_.regularization * static_cast<double>(n_fit_rows_));
  }

  int numRows() const { return n_fit_rows_ + 2; }
  int numFitRows() const { return n_fit_rows_; }
  void setFlatness(const double f) { flatness_ = f; }

  IRL::Normal normalAt(const std::array<double, 2>& p) const {
    const double angle = std::hypot(p[0], p[1]);
    if (angle <= 0.0) return n0_;
    IRL::Normal axis = p[0] * t0_ + p[1] * t1_;
    axis.normalize();
    return detail::rotate(n0_, axis, angle);
  }

  IRL::PlanarSeparator build(const std::array<double, 2>& p) const {
    const IRL::Normal nrm = normalAt(p);
    IRL::PlanarSeparator sep =
        IRL::PlanarSeparator::fromOnePlane(IRL::Plane(nrm, nrm * pivot_));
    IRL::setDistanceToMatchVolumeFraction(stencil_[0].cell, stencil_[0].vf,
                                          &sep, options_.volume_tolerance);
    return sep;
  }

  std::vector<double> residual(const std::array<double, 2>& p) const {
    const IRL::PlanarSeparator sep = build(p);
    std::vector<double> res(numRows(), 0.0);
    int row = 0;
    for (const auto& sc : stencil_) {
      const auto m = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(sc.cell, sep);
      res[row++] = m[0].volume() / sc.volume - sc.vf;
      for (int d = 0; d < 3; ++d)
        res[row++] = sc.liquid_meaningful
                         ? (m[0].centroid()[d] - sc.liquid_bary[d]) / width_
                         : 0.0;
      for (int d = 0; d < 3; ++d)
        res[row++] = sc.gas_meaningful
                         ? (m[1].centroid()[d] - sc.gas_bary[d]) / width_
                         : 0.0;
    }
    const double w = reg_base_ * flatness_;
    res[row++] = w * p[0];
    res[row++] = w * p[1];
    return res;
  }

  void clamp(std::array<double, 2>& p) const {
    const double tilt = std::hypot(p[0], p[1]);
    if (tilt > options_.max_rotation) {
      p[0] *= options_.max_rotation / tilt;
      p[1] *= options_.max_rotation / tilt;
    }
  }

 private:
  const std::vector<StencilCell>& stencil_;
  Options options_;
  IRL::Normal n0_, t0_, t1_;
  IRL::Pt pivot_;
  double width_ = 1.0, reg_base_ = 0.0, flatness_ = 1.0;
  int n_fit_rows_ = 0;
};

// Levenberg-Marquardt, Madsen-Nielsen damping; 2 params so the damped normal
// equations go through Cramer's rule, matching the Fortran exactly.
inline int runLM(const PlaneFit& model, std::array<double, 2>& p,
                 const Options& opt, double* fit_norm) {
  auto dot = [](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
  };
  auto fitNorm = [&model](const std::vector<double>& v) {
    double s = 0.0;
    for (int i = 0; i < model.numFitRows(); ++i) s += v[i] * v[i];
    return std::sqrt(s);
  };

  const int nrow = model.numRows();
  std::array<double, 2> p_try;
  std::vector<double> r = model.residual(p), r_try;
  double cost = 0.5 * dot(r, r);
  double damping = -1.0, growth = 2.0;
  int ite = 0;

  for (; ite < opt.max_ite; ++ite) {
    std::vector<double> jac(static_cast<std::size_t>(nrow) * 2);
    for (int d = 0; d < 2; ++d) {
      p_try = p;
      p_try[d] += opt.fd_step;
      const std::vector<double> rp = model.residual(p_try);
      for (int i = 0; i < nrow; ++i)
        jac[i * 2 + d] = (rp[i] - r[i]) / opt.fd_step;
    }

    double g[2] = {0.0, 0.0};
    double h[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int i = 0; i < nrow; ++i)
      for (int a = 0; a < 2; ++a) {
        g[a] += jac[i * 2 + a] * r[i];
        for (int b = 0; b < 2; ++b) h[a][b] += jac[i * 2 + a] * jac[i * 2 + b];
      }

    const double maxdiag = std::max(h[0][0], h[1][1]);
    if (std::max(std::abs(g[0]), std::abs(g[1])) < 1.0e-11) break;
    if (damping < 0.0) damping = 1.0e-3 * maxdiag;

    bool stepped = false;
    for (int trial = 0; trial < opt.max_trial && !stepped; ++trial) {
      double s[2][2] = {{h[0][0], h[0][1]}, {h[1][0], h[1][1]}};
      for (int d = 0; d < 2; ++d)
        s[d][d] += damping * std::max(h[d][d], 1.0e-10 * maxdiag);
      const double det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
      if (std::abs(det) <= std::numeric_limits<double>::min()) {
        damping *= growth; growth *= 2.0; continue;
      }
      const double step[2] = {(-g[0] * s[1][1] + g[1] * s[0][1]) / det,
                              (-g[1] * s[0][0] + g[0] * s[1][0]) / det};
      if (std::hypot(step[0], step[1]) < 1.0e-11 * (std::hypot(p[0], p[1]) + 1.0e-11)) {
        stepped = true;
        break;
      }

      p_try = {p[0] + step[0], p[1] + step[1]};
      model.clamp(p_try);
      r_try = model.residual(p_try);
      const double cost_try = 0.5 * dot(r_try, r_try);
      const double predicted = 0.5 * (step[0] * (damping * step[0] - g[0]) +
                                      step[1] * (damping * step[1] - g[1]));
      const double gain = predicted > 0.0 ? (cost - cost_try) / predicted : -1.0;
      if (gain <= 0.0) { damping *= growth; growth *= 2.0; continue; }

      p = p_try; r = r_try; cost = cost_try;
      const double q = 2.0 * gain - 1.0;
      damping *= std::max(1.0 / 3.0, 1.0 - q * q * q);
      growth = 2.0;
      stepped = true;
    }
    if (!stepped) break;
    if (fitNorm(r) < 1.0e-9) break;
  }

  *fit_norm = fitNorm(r);
  return ite;
}

// Literal transcription of the Fortran is_locally_flat. planes[0] is the
// center; planes[1..] each carry their own reconstructed plane.
inline bool gateFlatness(const std::vector<NeighborPlane>& planes,
                         const Options& opt, double* flatness) {
  *flatness = 1.0;
  const NeighborPlane& center = planes[0];
  if (!center.mixed) return false;

  double worst_dot = 1.0;
  int nnbr = 0;
  for (std::size_t m = 1; m < planes.size(); ++m) {
    const auto& nb = planes[m];
    if (!nb.mixed) continue;   // pure cell: no interface polygon

    const auto bary =
        plicgeom::planeBoxPolygonCentroid(nb.lo, nb.hi, nb.normal, nb.distance);
    if (!bary) continue;

    // Orientation must agree with ours
    const double d = nb.normal * center.normal;
    if (d < opt.flat_mindot) return false;
    // ...and the neighbor's interface must sit on our own plane (width = 1)
    if (std::abs(center.normal * (*bary) - center.distance) > opt.flat_maxdev)
      return false;

    worst_dot = std::min(worst_dot, d);
    ++nnbr;
  }

  // Too little surrounding interface to judge: treat as not flat
  if (nnbr < opt.flat_minnbr) return false;
  *flatness = 0;//(1.0 - worst_dot) / (1.0);// - opt.flat_mindot);
  return true;
}

// Refines the center cell's normal. On gate rejection, `normal` is left at
// the network's own prediction (planes[0].normal).
inline Result refine(const std::vector<StencilCell>& fit_stencil,
                     const std::vector<NeighborPlane>& planes,
                     IRL::Normal& normal, const Options& opt = Options()) {
  Result out;
  normal = planes[0].normal;

  if (!gateFlatness(planes, opt, &out.flatness)) return out;

  PlaneFit model(fit_stencil, normal, opt);
  model.setFlatness(out.flatness);
  std::array<double, 2> p = {0.0, 0.0};
  model.clamp(p);
  out.iterations = runLM(model, p, opt, &out.residual);
  normal = model.normalAt(p);
  out.fitted = true;
  return out;
}

// Surface points for the paraboloid fit: each mixed cell's own reconstructed
// polygon, with centroid, normal, and area. Center cell first -- it defines
// the reference point and frame for the fit.
// Reconstructed interface polygons for the paraboloid fits: each mixed cell's
// own plane cut against its own cell, with vertices, centroid, normal, area.
// Center cell first -- it defines the reference point and frame.
inline std::vector<plicparab::SurfacePolygon> gatherSurfacePolygons(
    const std::vector<NeighborPlane>& planes) {
  std::vector<plicparab::SurfacePolygon> polys;
  polys.reserve(planes.size());
  for (std::size_t m = 0; m < planes.size(); ++m) {
    const auto& nb = planes[m];
    if (!nb.mixed) {
      if (m == 0) return {};
      continue;
    }
    const auto info =
        plicgeom::planeBoxPolygon(nb.lo, nb.hi, nb.normal, nb.distance);
    if (!info) {
      if (m == 0) return {};
      continue;
    }
    plicparab::SurfacePolygon sp;
    sp.vertices = info->vertices;
    sp.centroid = info->centroid;
    sp.normal = nb.normal;
    sp.area = info->area;
    polys.push_back(sp);
  }
  return polys;
}

// Orientation from a local paraboloid fitted to the neighborhood's
// reconstructed interface polygons. With parab_use_gate, the same flatness
// gate as the plane branch runs first, so the two methods are compared on
// identical cell populations. The incoming `normal` (the network's
// prediction) is kept on rejection and resolves the sign on success.
inline Result refineParaboloid(const std::vector<NeighborPlane>& planes,
                               IRL::Normal& normal,
                               const Options& opt = Options()) {
  Result out;
  const IRL::Normal seed = normal;

  double gate_flatness = 1.0;
  if (opt.parab_use_gate) {
    if (!gateFlatness(planes, opt, &gate_flatness)) return out;
  }

const auto polys = gatherSurfacePolygons(planes);
  if (static_cast<int>(polys.size()) < opt.parab_minpts) return out;

  const auto fit = (opt.orientation_method == 2)
      ? plicparab::fitIntegral(polys, 1.0, opt.parab_h)
      : plicparab::fitPointwise(polys, 1.0, opt.parab_h);
  if (!fit) return out;
  if (fit->rms_residual > opt.parab_maxresid) return out;

  IRL::Normal fitted = fit->normal;
  if (fitted * seed < 0.0) fitted = -fitted;

  normal = fitted;
  out.fitted = true;
  out.residual = fit->rms_residual;
  // Fitted mean curvature, not the gate's dot-product margin -- a direct
  // measurement rather than a proxy. gate_flatness is available above if you
  // want the two side by side.
  out.flatness = std::abs(fit->curvature);
  return out;
}

}  // namespace plicfit

#endif  // PLIC_REFINE_H_