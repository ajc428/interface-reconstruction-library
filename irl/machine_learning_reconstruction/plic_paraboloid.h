#ifndef PLIC_PARABOLOID_H_
#define PLIC_PARABOLOID_H_

#include <Eigen/Dense>

#include <cmath>
#include <optional>
#include <vector>

#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"

// Local paraboloid fits of the IRL surface, ported from the Fortran
// paraboloid_fit (pointwise) and paraboloid_integral_fit (integral), reduced
// to a 3x3x3 support.
//
// Both fit, in a frame anchored to the center polygon,
//   n = F(t,s) = b1 + b2 t + b3 s + b4 t^2 + b5 t s + b6 s^2
// where n runs along the center polygon's normal and (t,s) span its tangent
// plane. This is Eq. (1) of the paraboloid formulation: a,b,c,d,e,f.
//
// NOTE ON BASIS ORDERING: the Fortran paraboloid_fit uses a different basis,
// [1, t, s, t^2/2, s^2/2, t s], which puts s^2 and ts in swapped slots and
// folds in the 1/2 factors. paraboloid_integral_fit, Jibben_3D::solve,
// fitParaboloidToPLICHeights and the paper all use [1, t, s, t^2, ts, s^2].
// This header uses the latter throughout so both fits share the downstream
// derivative extraction and canonical decomposition; the pointwise assembly
// below is adjusted accordingly (no 1/2 factors, second derivatives carry 2x).
namespace plicparab {

// One neighbor polygon's contribution.
struct SurfacePolygon {
  std::vector<IRL::Pt> vertices;   // ordered; required by the integral fit
  IRL::Pt centroid;
  IRL::Normal normal;
  double area = 0.0;               // used by the pointwise fit's weight only
};

struct FitResult {
  IRL::Normal normal;              // surface normal at (t,s) = (0,0)
  double curvature = 0.0;          // mean curvature at (t,s) = (0,0)
  double rms_residual = 0.0;
  int num_points = 0;
  std::array<double, 6> coeffs{};  // a,b,c,d,e,f in the frame below
  IRL::Normal nref, tref, sref;    // the frame the coeffs live in
};

// Quasi-Gaussian weight, h = 2.5 per the Fortran (and Jibben's delta).
inline double wgauss(const double d, const double h) {
  if (d >= h) return 0.0;
  const double r = d / h;
  const double om = 1.0 - r;
  return (1.0 + 4.0 * r) * om * om * om * om;
}

// Frame from the center polygon: same largest-component branch as both
// Fortran routines, to avoid a degenerate cross product.
inline void buildFrame(const IRL::Normal& n_in, IRL::Normal* nref,
                       IRL::Normal* tref, IRL::Normal* sref) {
  *nref = n_in;
  nref->normalize();
  const double a0 = std::abs((*nref)[0]), a1 = std::abs((*nref)[1]),
               a2 = std::abs((*nref)[2]);
  if (a0 >= a1 && a0 >= a2) {
    *tref = IRL::Normal((*nref)[1], -(*nref)[0], 0.0);
  } else if (a1 >= a2) {
    *tref = IRL::Normal(0.0, (*nref)[2], -(*nref)[1]);
  } else {
    *tref = IRL::Normal(-(*nref)[2], 0.0, (*nref)[0]);
  }
  tref->normalize();
  *sref = IRL::crossProduct(*nref, *tref);
  sref->normalize();
}

// Shared post-processing: derivatives at the origin, then normal + curvature.
// With the [1,t,s,t^2,ts,s^2] basis the second derivatives carry the 2x.
inline void finishFit(const Eigen::VectorXd& sol, const IRL::Normal& nref,
                      const IRL::Normal& tref, const IRL::Normal& sref,
                      const double mesh_size, FitResult* out) {
  const double dF_dt = sol(1);
  const double dF_ds = sol(2);
  const double ddF_dtdt = 2.0 * sol(3);
  const double ddF_dtds = sol(4);
  const double ddF_dsds = 2.0 * sol(5);

  const double denom = std::pow(1.0 + dF_dt * dF_dt + dF_ds * dF_ds, 1.5);
  out->curvature = -((1.0 + dF_dt * dF_dt) * ddF_dsds -
                     2.0 * dF_dt * dF_ds * ddF_dtds +
                     (1.0 + dF_ds * dF_ds) * ddF_dtdt) / denom;
  out->curvature /= mesh_size;

  // Surface point is pref + t*tref + s*sref + F*nref, so the tangents give
  // normal proportional to nref - F_t*tref - F_s*sref.
  IRL::Normal fitted = nref - dF_dt * tref - dF_ds * sref;
  fitted.normalize();
  out->normal = fitted;

  for (int i = 0; i < 6; ++i) out->coeffs[i] = sol(i);
  out->nref = nref;
  out->tref = tref;
  out->sref = sref;
}

// ---------------------------------------------------------------------------
// Pointwise fit (Fortran paraboloid_fit)
// ---------------------------------------------------------------------------
//
// One row per neighbor polygon centroid, weighted by area x alignment x
// distance kernel. polygons[0] must be the center cell's polygon.
inline std::optional<FitResult> fitPointwise(
    const std::vector<SurfacePolygon>& polygons, const double mesh_size = 1.0,
    const double h = 2.5) {
  if (polygons.size() < 6) return std::nullopt;

  const IRL::Pt pref = polygons[0].centroid;
  IRL::Normal nref, tref, sref;
  buildFrame(polygons[0].normal, &nref, &tref, &sref);

  std::vector<std::array<double, 6>> rows;
  std::vector<double> rhs;
  for (const auto& sp : polygons) {
    IRL::Normal nloc = sp.normal;
    nloc.normalize();

    const IRL::Pt d((sp.centroid[0] - pref[0]) / mesh_size,
                    (sp.centroid[1] - pref[1]) / mesh_size,
                    (sp.centroid[2] - pref[2]) / mesh_size);
    const double pn = nref * d, pt = tref * d, ps = sref * d;

    const double surf = sp.area / (mesh_size * mesh_size);
    const double align = std::max(nloc * nref, 0.0);
    const double dist = std::sqrt(pn * pn + pt * pt + ps * ps);
    const double ww = surf * align * wgauss(dist, h);
    if (ww <= 0.0) continue;

    const double sw = std::sqrt(ww);
    rows.push_back({sw, sw * pt, sw * ps, sw * pt * pt, sw * pt * ps,
                    sw * ps * ps});
    rhs.push_back(sw * pn);
  }

  const int ndata = static_cast<int>(rows.size());
  if (ndata < 6) return std::nullopt;

  Eigen::MatrixXd A(ndata, 6);
  Eigen::VectorXd b(ndata);
  for (int i = 0; i < ndata; ++i) {
    for (int j = 0; j < 6; ++j) A(i, j) = rows[i][j];
    b(i) = rhs[i];
  }
  const Eigen::VectorXd sol =
      A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

  FitResult out;
  out.num_points = ndata;
  out.rms_residual =
      (A * sol - b).norm() / std::sqrt(static_cast<double>(ndata));
  finishFit(sol, nref, tref, sref, mesh_size, &out);
  return out;
}

// ---------------------------------------------------------------------------
// Integral fit (Fortran paraboloid_integral_fit / Jibben)
// ---------------------------------------------------------------------------
//
// One equation per neighbor polygon: the integral of the fitted quadratic
// over that polygon's projected footprint must equal the integral of the
// polygon's OWN plane over the same footprint. Each polygon therefore
// contributes its full extent rather than a single point, which is better
// conditioned on coarse interfaces where a centroid is a poor summary.
//
// No area or alignment weight here: the monomial integrals already carry
// area, and back-facing polygons are hard-skipped (as in the Fortran and in
// Jibben_3D) rather than softly down-weighted.
inline std::optional<FitResult> fitIntegral(
    const std::vector<SurfacePolygon>& polygons, const double mesh_size = 1.0,
    const double h = 2.5) {
  if (polygons.size() < 6) return std::nullopt;

  const IRL::Pt pref = polygons[0].centroid;
  IRL::Normal nref, tref, sref;
  buildFrame(polygons[0].normal, &nref, &tref, &sref);

  Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
  int ndata = 0;

  // Kept to report a residual comparable to the pointwise fit's.
  std::vector<Eigen::Matrix<double, 6, 1>> kept_rows;
  std::vector<double> kept_rhs;

  for (const auto& sp : polygons) {
    const int shape = static_cast<int>(sp.vertices.size());
    if (shape < 3) continue;

    IRL::Normal nglob = sp.normal;
    nglob.normalize();
    // Skip polygons oriented more than 90 degrees from the center normal.
    if (nglob * nref <= 0.0) continue;

    // Local frame components, ordered (t, s, n) to match the Fortran.
    const IRL::Normal nloc(tref * nglob, sref * nglob, nref * nglob);
    // Guard the near-perpendicular case: the plane cannot be written as a
    // height field over (t,s) and the coefficients below blow up.
    if (std::abs(nloc[2]) < 1.0e-10) continue;

    const IRL::Pt dc((sp.centroid[0] - pref[0]) / mesh_size,
                     (sp.centroid[1] - pref[1]) / mesh_size,
                     (sp.centroid[2] - pref[2]) / mesh_size);
    const IRL::Pt ploc(tref * dc, sref * dc, nref * dc);

    // Polygon's own plane as n = c0 + c1 t + c2 s.
    const double ndotp =
        nloc[0] * ploc[0] + nloc[1] * ploc[1] + nloc[2] * ploc[2];
    const double inv = 1.0 / (-nloc[2]);
    const double c0 = -ndotp * inv;
    const double c1 = nloc[0] * inv;
    const double c2 = nloc[1] * inv;

    // Monomial integrals over the projected footprint, by Green's theorem on
    // the polygon boundary: [1, t, s, t^2, ts, s^2].
    Eigen::Matrix<double, 6, 1> integrals = Eigen::Matrix<double, 6, 1>::Zero();
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
                      (3.0 * xv * xv * yv + xv * xv * yvn +
                       2.0 * xv * xvn * yv + 2.0 * xv * xvn * yvn +
                       xvn * xvn * yv + 3.0 * xvn * xvn * yvn) / 24.0;
      integrals(5) += (xv - xvn) * (yv + yvn) * (yv * yv + yvn * yvn) / 12.0;
    }

    const double b_dot_sum =
        c0 * integrals(0) + c1 * integrals(1) + c2 * integrals(2);

    const double dist = std::sqrt(ploc[0] * ploc[0] + ploc[1] * ploc[1] +
                                  ploc[2] * ploc[2]);
    const double ww = wgauss(dist, h);
    if (ww <= 0.0) continue;

    A += ww * integrals * integrals.transpose();
    b += ww * integrals * b_dot_sum;
    kept_rows.push_back(std::sqrt(ww) * integrals);
    kept_rhs.push_back(std::sqrt(ww) * b_dot_sum);
    ++ndata;
  }

  if (ndata < 6) return std::nullopt;

  // Symmetric solve (the Fortran uses dsysv); fall back to QR if indefinite.
  Eigen::Matrix<double, 6, 1> sol6 = A.ldlt().solve(b);
  if (!sol6.allFinite()) sol6 = A.colPivHouseholderQr().solve(b);
  if (!sol6.allFinite()) return std::nullopt;
  const Eigen::VectorXd sol = sol6;

  FitResult out;
  out.num_points = ndata;
  double s2 = 0.0;
  for (int i = 0; i < ndata; ++i) {
    const double r = kept_rows[i].dot(sol6) - kept_rhs[i];
    s2 += r * r;
  }
  out.rms_residual = std::sqrt(s2 / static_cast<double>(ndata));
  finishFit(sol, nref, tref, sref, mesh_size, &out);
  return out;
}

// ---------------------------------------------------------------------------
// Canonical decomposition (Eqs. 2-5 / Jibben_3D::solve)
// ---------------------------------------------------------------------------
//
// Principal curvatures A,B, in-plane rotation theta, and the apex offset
// (u,v,w) in the (t,s,n) frame that takes Eq. (1) to Eq. (2). Formulas taken
// verbatim from Jibben_3D::solve rather than re-derived.
struct Canonical {
  double A = 0.0, B = 0.0, theta = 0.0;
  double u = 0.0, v = 0.0, w = 0.0;
  bool apex_valid = false;   // false if 4df - e^2 is degenerate (plane/cylinder)
};

inline Canonical decompose(const FitResult& fit) {
  const double a = fit.coeffs[0], b = fit.coeffs[1], c = fit.coeffs[2],
               d = fit.coeffs[3], e = fit.coeffs[4], f = fit.coeffs[5];
  Canonical out;
  const double dmf = d - f;
  out.theta = 0.5 * std::atan2(e, std::abs(dmf) < 1.0e-15
                                      ? std::copysign(1.0e-15, dmf) : dmf);
  const double ct = std::cos(out.theta), st = std::sin(out.theta);
  out.A = -(d * ct * ct + f * st * st + e * ct * st);
  out.B = -(f * ct * ct + d * st * st - e * ct * st);

  const double denom = 4.0 * d * f - e * e;
  if (std::abs(denom) > 1.0e-15) {
    const double di = 1.0 / denom;
    out.u = (2.0 * b * f - c * e) * di;
    out.v = -(b * e - 2.0 * d * c) * di;
    out.w = -(a + (-b * b * f + b * c * e - c * c * d) * di);
    out.apex_valid = true;
  }
  return out;
}

}  // namespace plicparab

#endif  // PLIC_PARABOLOID_H_