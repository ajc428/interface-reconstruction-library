// Pass 2 of R2P3D_NetFast: r2p_paraboloid_pass.h with its default options
// fixed, fixed-size storage instead of per-cell vectors, a QR least-squares
// solve instead of an SVD (the ridge rows make the system full rank, so the
// solution is the same), and no copy of the whole field: every cell reads the
// pass-1 reconstruction and the results are written back after the sweep.
//
// For each two-plane cell: take the 3^3 stencil's interface polygons, sort
// them into the film's two faces by orientation, fit both faces jointly as
// height fields over one frame (shared shape, separate offsets, ridged
// splay), rotate the normals toward the fit (at most kMaxRotation), re-solve
// the distances, and keep one plane instead if it matches the film centroid
// about as well.

#ifndef EXAMPLES_NEW_ADVECTOR_R2P_FAST_PARAB_H_
#define EXAMPLES_NEW_ADVECTOR_R2P_FAST_PARAB_H_

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/machine_learning_reconstruction/plic_paraboloid.h"
#include "irl/moments/separated_volume_moments.h"
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"
#include "examples/new_advector/r2p_newton_distance.h"
#include "examples/new_advector/r2p_snap.h"
#include "examples/new_advector/r2p_nopinch.h"

namespace r2pfastparab {

constexpr double kMaxRotation = 0.35;       // rad, per normal
constexpr double kMinGroupAreaFraction = 0.10;
constexpr double kH = 2.5;                  // Gaussian weight radius, cells
constexpr double kSplayPenalty = 0.015;
constexpr double kCurvaturePenalty = 1.0e-3;
constexpr int kMinPerGroup = 6;
constexpr double kMaxResid = 0.25;
constexpr double kMinBisector = 1.0e-3;
constexpr double kMaxSplayChange = 0.30;    // rad
constexpr double kPlaneDropBias = 1.05;

struct Poly {
  IRL::Pt centroid;
  IRL::Normal normal;
  double area = 0.0;
  int group = -1;
  bool is_center = false;
};

// Plane p of `sep` inside the box [lo,hi], clipped to the part that bounds
// the phases (below the other planes; above them when flipped).
inline bool polygonFor(const IRL::Pt& lo, const IRL::Pt& hi,
                       const IRL::PlanarSeparator& sep,
                       const IRL::UnsignedIndex_t p, Poly* out) {
  const IRL::Normal& normal = sep[p].normal();
  const double distance = sep[p].distance();
  const double vx[2] = {lo[0], hi[0]}, vy[2] = {lo[1], hi[1]}, vz[2] = {lo[2], hi[2]};
  double vpx[8], vpy[8], vpz[8], s[8];
  int n = 0;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c) {
        vpx[n] = vx[a]; vpy[n] = vy[b]; vpz[n] = vz[c];
        s[n] = normal[0] * vpx[n] + normal[1] * vpy[n] + normal[2] * vpz[n] - distance;
        ++n;
      }
  static const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                   {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  std::array<IRL::Pt, 12> pts;
  int np = 0;
  for (const auto& e : edges) {
    const double s0 = s[e[0]], s1 = s[e[1]];
    if ((s0 <= 0.0 && s1 >= 0.0) || (s0 >= 0.0 && s1 <= 0.0)) {
      if (std::abs(s0 - s1) < 1.0e-14) continue;
      const double t = s0 / (s0 - s1);
      pts[np++] = IRL::Pt(vpx[e[0]] + t * (vpx[e[1]] - vpx[e[0]]),
                          vpy[e[0]] + t * (vpy[e[1]] - vpy[e[0]]),
                          vpz[e[0]] + t * (vpz[e[1]] - vpz[e[0]]));
    }
  }
  if (np < 3) return false;

  // Order the points around their mean.
  IRL::Normal t0 = IRL::crossProduct(
      normal, std::abs(normal[0]) < 0.9 ? IRL::Normal(1, 0, 0) : IRL::Normal(0, 1, 0));
  t0.normalize();
  IRL::Normal t1 = IRL::crossProduct(normal, t0);
  t1.normalize();
  double ax = 0.0, ay = 0.0, az = 0.0;
  for (int i = 0; i < np; ++i) { ax += pts[i][0]; ay += pts[i][1]; az += pts[i][2]; }
  ax /= static_cast<double>(np);
  ay /= static_cast<double>(np);
  az /= static_cast<double>(np);
  std::array<std::pair<double, int>, 12> order;
  for (int i = 0; i < np; ++i) {
    const IRL::Pt d(pts[i][0] - ax, pts[i][1] - ay, pts[i][2] - az);
    order[i] = {std::atan2(t1 * d, t0 * d), i};
  }
  std::sort(order.begin(), order.begin() + np);
  double area2 = 0.0;
  for (int i = 0; i < np; ++i) {
    const IRL::Pt& p0 = pts[order[i].second];
    const IRL::Pt& p1 = pts[order[(i + 1) % np].second];
    const double e1x = p0[0] - ax, e1y = p0[1] - ay, e1z = p0[2] - az;
    const double e2x = p1[0] - ax, e2y = p1[1] - ay, e2z = p1[2] - az;
    const double crx = e1y * e2z - e1z * e2y, cry = e1z * e2x - e1x * e2z,
                 crz = e1x * e2y - e1y * e2x;
    area2 += std::sqrt(crx * crx + cry * cry + crz * crz);
  }
  if (area2 < 1.0e-30) return false;

  // Sutherland-Hodgman clip against every other plane.
  std::array<IRL::Pt, 16> buf[2];
  int nv = np;
  for (int i = 0; i < np; ++i) buf[0][i] = pts[order[i].second];
  int cur = 0;
  const double side = sep.isFlipped() ? -1.0 : 1.0;
  for (IRL::UnsignedIndex_t q = 0; q < sep.getNumberOfPlanes(); ++q) {
    if (q == p) continue;
    const IRL::Normal cn = side * sep[q].normal();
    const double cd = side * sep[q].distance();
    const std::array<IRL::Pt, 16>& in = buf[cur];
    std::array<IRL::Pt, 16>& o = buf[1 - cur];
    int no = 0;
    for (int i = 0; i < nv; ++i) {
      const IRL::Pt& a = in[i];
      const IRL::Pt& b = in[(i + 1) % nv];
      const double dc = cn * a - cd, dn = cn * b - cd;
      if (dc <= 0.0 && no < 16) o[no++] = a;
      if (((dc < 0.0 && dn > 0.0) || (dc > 0.0 && dn < 0.0)) && no < 16) {
        const double t = dc / (dc - dn);
        o[no++] = IRL::Pt(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
                          a[2] + t * (b[2] - a[2]));
      }
    }
    nv = no;
    cur = 1 - cur;
    if (nv < 3) return false;
  }

  // Fan area and centroid from vertex 0.
  const std::array<IRL::Pt, 16>& v = buf[cur];
  double total = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
  for (int i = 1; i + 1 < nv; ++i) {
    const double e1x = v[i][0] - v[0][0], e1y = v[i][1] - v[0][1], e1z = v[i][2] - v[0][2];
    const double e2x = v[i + 1][0] - v[0][0], e2y = v[i + 1][1] - v[0][1],
                 e2z = v[i + 1][2] - v[0][2];
    const double crx = e1y * e2z - e1z * e2y, cry = e1z * e2x - e1x * e2z,
                 crz = e1x * e2y - e1y * e2x;
    const double tri = 0.5 * std::sqrt(crx * crx + cry * cry + crz * crz);
    if (tri <= 0.0) continue;
    total += tri;
    cx += tri * (v[0][0] + v[i][0] + v[i + 1][0]) / 3.0;
    cy += tri * (v[0][1] + v[i][1] + v[i + 1][1]) / 3.0;
    cz += tri * (v[0][2] + v[i][2] + v[i + 1][2]) / 3.0;
  }
  if (total <= 0.0) return false;
  out->centroid = IRL::Pt(cx / total, cy / total, cz / total);
  out->normal = normal;
  out->normal.normalize();
  out->area = total;
  return true;
}

// Rotates `from` toward `to` by at most `max_angle` (Rodrigues).
inline IRL::Normal limitedRotation(const IRL::Normal& from, const IRL::Normal& to,
                                   const double max_angle) {
  const double angle = std::acos(std::max(-1.0, std::min(1.0, from * to)));
  const double target = std::min(angle, max_angle);
  if (angle < 1.0e-12 || target < 1.0e-12) return from;
  IRL::Normal axis = IRL::crossProduct(from, to);
  if (axis.calculateMagnitude() < 1.0e-12) return from;
  axis.normalize();
  const double c = std::cos(target), s = std::sin(target);
  IRL::Normal out = from * c + IRL::crossProduct(axis, from) * s + axis * (axis * from) * (1.0 - c);
  out.normalize();
  return out;
}

// Joint fit of the two faces: face g is n = a0_g + (a1+d1_g) t + (a2+d2_g) s
// + a3 t^2 + a4 ts + a5 s^2 over the bisector frame. polys[g][0] must be the
// centre cell's polygon. Returns false if the fit is rejected.
inline bool fitCoupled(const Poly* const polys[2], const int npoly[2],
                       const IRL::Normal seed[2], const double mesh_size,
                       IRL::Normal fitted[2]) {
  if (npoly[0] < kMinPerGroup || npoly[1] < kMinPerGroup) return false;

  IRL::Pt pref = polys[0][0].centroid;
  IRL::Normal seed_normal = polys[0][0].normal;
  {
    IRL::Normal n0 = polys[0][0].normal, n1 = polys[1][0].normal;
    n0.normalize();
    n1.normalize();
    const double w0 = 0.5, w1 = 0.5;
    const IRL::Normal combined = w0 * n0 - w1 * n1;
    if (combined.calculateMagnitude() >= kMinBisector) {
      seed_normal = combined;
      pref = IRL::Pt(w0 * polys[0][0].centroid[0] + w1 * polys[1][0].centroid[0],
                     w0 * polys[0][0].centroid[1] + w1 * polys[1][0].centroid[1],
                     w0 * polys[0][0].centroid[2] + w1 * polys[1][0].centroid[2]);
    }
  }
  IRL::Normal nref, tref, sref;
  plicparab::buildFrame(seed_normal, &nref, &tref, &sref);

  // [a0_0, a0_1, a1, a2, a3, a4, a5, d1_0, d2_0, d1_1, d2_1]
  constexpr int kN = 11, kMaxRows = 64;
  using Mat = Eigen::Matrix<double, Eigen::Dynamic, kN, Eigen::ColMajor, kMaxRows, kN>;
  using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, kMaxRows, 1>;
  Mat A = Mat::Zero(kMaxRows, kN);
  Vec b = Vec::Zero(kMaxRows);
  int ndata = 0, count[2] = {0, 0};
  double weight_total = 0.0;
  for (int g = 0; g < 2; ++g) {
    const IRL::Normal facing = (g == 0) ? nref : -nref;
    for (int idx = 0; idx < npoly[g]; ++idx) {
      const Poly& sp = polys[g][idx];
      IRL::Normal nglob = sp.normal;
      nglob.normalize();
      const IRL::Pt dc((sp.centroid[0] - pref[0]) / mesh_size,
                       (sp.centroid[1] - pref[1]) / mesh_size,
                       (sp.centroid[2] - pref[2]) / mesh_size);
      const double pt = tref * dc, ps = sref * dc, pn = nref * dc;
      const double wg = plicparab::wgauss(std::sqrt(pt * pt + ps * ps + pn * pn), kH);
      if (wg <= 0.0) continue;
      const double w = sp.area / (mesh_size * mesh_size) * std::max(nglob * facing, 0.0) * wg;
      if (w <= 0.0 || ndata >= kMaxRows - 7) continue;
      const double sw = std::sqrt(w);
      A(ndata, g) = sw;
      A(ndata, 2) = sw * pt;
      A(ndata, 3) = sw * ps;
      A(ndata, 4) = sw * pt * pt;
      A(ndata, 5) = sw * pt * ps;
      A(ndata, 6) = sw * ps * ps;
      A(ndata, 7 + 2 * g) = sw * pt;
      A(ndata, 8 + 2 * g) = sw * ps;
      b(ndata) = sw * pn;
      weight_total += w;
      ++count[g];
      ++ndata;
    }
  }
  if (count[0] < kMinPerGroup || count[1] < kMinPerGroup) return false;

  const int ntotal = ndata + 7;
  const double splay_w = std::sqrt(kSplayPenalty * weight_total);
  const double curv_w = std::sqrt(kCurvaturePenalty * weight_total);
  int prow = ndata;
  for (int c = 7; c < kN; ++c) A(prow++, c) = splay_w;
  for (int c = 4; c < 7; ++c) A(prow++, c) = curv_w;
  const Mat At = A.topRows(ntotal);
  const Eigen::Matrix<double, kN, 1> sol = At.colPivHouseholderQr().solve(b.head(ntotal));
  if (!sol.allFinite()) return false;

  double s2 = 0.0;
  for (int i = 0; i < ndata; ++i) {
    const double res = A.row(i).dot(sol.transpose()) - b(i);
    s2 += res * res;
  }
  if (std::sqrt(s2 / static_cast<double>(ndata)) > kMaxResid) return false;

  for (int g = 0; g < 2; ++g) {
    const double ft = sol(2) + sol(7 + 2 * g), fs = sol(3) + sol(8 + 2 * g);
    IRL::Normal f = nref - ft * tref - fs * sref;
    f.normalize();
    if (f * seed[g] < 0.0) f = -f;
    fitted[g] = f;
  }
  return true;
}

inline double centroidError(const IRL::RectangularCuboid& cell, const IRL::PlanarSeparator& sep,
                            const int phase, const IRL::Pt& target) {
  const auto moments =
      IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>,
                                      IRL::ReconstructionDefaultCuttingMethod>(cell, sep);
  const IRL::Pt c = moments[phase].centroid();
  const double dx = c[0] - target[0], dy = c[1] - target[1], dz = c[2] - target[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// New normals for the two-plane cell (i,j,k), or false to keep pass 1's.
inline bool refineNormals(const Data<double>& vf, const Data<IRL::PlanarSeparator>& iface,
                          const int i, const int j, const int k, const double mesh_size,
                          IRL::Normal normal[2]) {
  const BasicMesh& mesh = vf.getMesh();
  std::array<Poly, 54> tagged;
  std::array<int, 28> cell_begin;
  int nt = 0, nc = 0;
  cell_begin[0] = 0;
  // Centre cell first, then the rest in stencil order.
  for (int pass = 0; pass < 2; ++pass)
    for (int ii = i - 1; ii <= i + 1; ++ii)
      for (int jj = j - 1; jj <= j + 1; ++jj)
        for (int kk = k - 1; kk <= k + 1; ++kk) {
          const bool is_center = (ii == i && jj == j && kk == k);
          if ((pass == 0) != is_center) continue;
          const IRL::PlanarSeparator& sep = iface(ii, jj, kk);
          const double f = vf(ii, jj, kk);
          if (f > IRL::global_constants::VF_LOW && f < IRL::global_constants::VF_HIGH) {
            const IRL::Pt lo(mesh.x(ii), mesh.y(jj), mesh.z(kk));
            const IRL::Pt hi(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1));
            for (IRL::UnsignedIndex_t p = 0; p < sep.getNumberOfPlanes() && nt < 54; ++p) {
              if (!polygonFor(lo, hi, sep, p, &tagged[nt])) continue;
              tagged[nt].group = -1;
              tagged[nt].is_center = is_center;
              ++nt;
            }
          }
          cell_begin[++nc] = nt;
        }
  if (nt < 2 * kMinPerGroup) return false;

  // Sort polygons into the faces by orientation; a cell's two planes jointly.
  const IRL::Normal net0 = normal[0], net1 = normal[1];
  auto single = [&](Poly& t) {
    const double d0 = t.normal * net0, d1 = t.normal * net1;
    t.group = (std::max(d0, d1) > 0.0) ? ((d0 >= d1) ? 0 : 1) : -1;
  };
  for (int c = 0; c < nc; ++c) {
    const int begin = cell_begin[c], end = cell_begin[c + 1];
    if (end - begin == 1) {
      single(tagged[begin]);
    } else if (end - begin >= 2) {
      Poly& a = tagged[begin];
      Poly& bb = tagged[begin + 1];
      const double a0 = a.normal * net0, a1 = a.normal * net1;
      const double b0 = bb.normal * net0, b1 = bb.normal * net1;
      if (a0 + b1 >= a1 + b0) {
        a.group = (a0 > 0.0) ? 0 : -1;
        bb.group = (b1 > 0.0) ? 1 : -1;
      } else {
        a.group = (a1 > 0.0) ? 1 : -1;
        bb.group = (b0 > 0.0) ? 0 : -1;
      }
      for (int extra = begin + 2; extra < end; ++extra) single(tagged[extra]);
    }
  }
  double area[2] = {0.0, 0.0}, total = 0.0;
  for (int t = 0; t < nt; ++t) {
    if (tagged[t].group >= 0) area[tagged[t].group] += tagged[t].area;
    total += tagged[t].area;
  }
  if (total <= 0.0) return false;
  if (std::min(area[0] / total, area[1] / total) < kMinGroupAreaFraction) return false;

  // Each face's polygons, centre-cell polygon first.
  std::array<Poly, 54> grouped[2];
  int ng[2] = {0, 0};
  for (int g = 0; g < 2; ++g) {
    for (int t = 0; t < nt; ++t)
      if (tagged[t].group == g && tagged[t].is_center) grouped[g][ng[g]++] = tagged[t];
    if (ng[g] == 0) return false;
    for (int t = 0; t < nt; ++t)
      if (tagged[t].group == g && !tagged[t].is_center) grouped[g][ng[g]++] = tagged[t];
  }
  const Poly* const polys[2] = {grouped[0].data(), grouped[1].data()};
  const IRL::Normal seed[2] = {net0, net1};
  IRL::Normal fit[2];
  if (!fitCoupled(polys, ng, seed, mesh_size, fit)) return false;

  const IRL::Normal limited0 = limitedRotation(net0, fit[0], kMaxRotation);
  const IRL::Normal limited1 = limitedRotation(net1, fit[1], kMaxRotation);
  const double splay_before = std::acos(std::max(-1.0, std::min(1.0, -(net0 * net1))));
  const double splay_after = std::acos(std::max(-1.0, std::min(1.0, -(limited0 * limited1))));
  if (std::abs(splay_after - splay_before) > kMaxSplayChange) return false;
  normal[0] = limited0;
  normal[1] = limited1;
  return true;
}

// Refines every two-plane cell listed in `cells` (interior indices). Reads the
// pass-1 field; writes after the sweep so the result is order-independent.
// Returns the cells it changed; `cell_seconds` (or nullptr) receives the time
// spent on each listed cell.
inline std::vector<std::array<int, 3>> run(const Data<double>& vf, const Data<IRL::Pt>& liq, const Data<IRL::Pt>& gas,
                const std::vector<std::array<int, 3>>& cells,
                Data<IRL::PlanarSeparator>* a_interface,
                std::vector<double>* cell_seconds = nullptr) {
  const BasicMesh& mesh = vf.getMesh();
  const double mesh_size = (mesh.dx() + mesh.dy() + mesh.dz()) / 3.0;
  std::vector<std::pair<std::array<int, 3>, IRL::PlanarSeparator>> updates;
  updates.reserve(cells.size());
  if (cell_seconds != nullptr) cell_seconds->assign(cells.size(), 0.0);
  for (std::size_t q = 0; q < cells.size(); ++q) {
    const auto& c = cells[q];
    const auto t0 = std::chrono::steady_clock::now();
    struct Stamp {   // records this cell's time on every exit from the loop body
      std::vector<double>* out;
      std::size_t q;
      std::chrono::steady_clock::time_point t0;
      ~Stamp() {
        if (out != nullptr)
          (*out)[q] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      }
    } stamp{cell_seconds, q, t0};
    const int i = c[0], j = c[1], k = c[2];
    const double f = vf(i, j, k);
    const IRL::PlanarSeparator& sep = (*a_interface)(i, j, k);
    if (sep.getNumberOfPlanes() != 2) continue;
    IRL::Normal normal[2] = {sep[0].normal(), sep[1].normal()};
    normal[0].normalize();
    normal[1].normalize();
    // A snapped thin film stays a slab: the fit may rotate it, not open it.
    const bool slab = r2psnap::isSlab(sep);
    if (!refineNormals(vf, *a_interface, i, j, k, mesh_size, normal)) continue;
    if (slab) r2psnap::keepSlab(normal[0], normal[1]);

    const double flip_i = sep.isNotFlipped() ? 1.0 : -1.0;
    const IRL::RectangularCuboid cube = IRL::RectangularCuboid::fromBoundingPts(
        IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
        IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));
    IRL::PlanarSeparator out = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(normal[0], 0.0), IRL::Plane(normal[1], 0.0), flip_i);
    one_plane_reason(i, j, k) = 0;
    r2pnewton::R2PNewtonDistanceSolver(f, liq(i, j, k), gas(i, j, k), out, cube);
    if (out.getNumberOfPlanes() != 2) one_plane_reason(i, j, k) = 7;
    r2pnopinch::applyAt(mesh, i, j, k, f, liq(i, j, k), gas(i, j, k), &out);
    if (one_plane_reason(i, j, k) == 0 && out.getNumberOfPlanes() != 2) one_plane_reason(i, j, k) = 5;

    // Keep one plane if it matches the film centroid about as well; never in
    // a thin film the guard holds for (r2p_edge_topology.h), nor in a very thin
    // film's PCA slab (r2p_snap.h pcaSlab).
    if (out.getNumberOfPlanes() == 2 && film_guard(i, j, k) == 0 && snapped(i, j, k) != 3) {
      const int film = flip_i < 0.0 ? 1 : 0;
      const IRL::Pt& target = film == 1 ? gas(i, j, k) : liq(i, j, k);
      const double err_two = centroidError(cube, out, film, target);
      double err_one = -1.0;
      IRL::PlanarSeparator best_one;
      for (int g = 0; g < 2; ++g) {
        IRL::PlanarSeparator cand = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal[g], 0.0));
        IRL::setDistanceToMatchVolumeFraction(cube, f, &cand);
        const double e = centroidError(cube, cand, film, target);
        if (err_one < 0.0 || e < err_one) {
          err_one = e;
          best_one = cand;
        }
      }
      if (err_one >= 0.0 && err_one <= kPlaneDropBias * err_two) {
        out = best_one;
        one_plane_reason(i, j, k) = 6;
      }
    }
    updates.emplace_back(c, out);
  }
  std::vector<std::array<int, 3>> changed;
  changed.reserve(updates.size());
  for (const auto& u : updates) {
    (*a_interface)(u.first[0], u.first[1], u.first[2]) = u.second;
    changed.push_back(u.first);
  }
  a_interface->updateBorder();
  return changed;
}

}  // namespace r2pfastparab

#endif  // EXAMPLES_NEW_ADVECTOR_R2P_FAST_PARAB_H_
