#ifndef PLIC_GEOMETRY_H_
#define PLIC_GEOMETRY_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"

// Centroid of the polygon formed by cutting an axis-aligned box with a plane.
// This is the offline stand-in for calculateCentroid(interface_polygon(...))
// in the Fortran gate: implemented directly so the harness doesn't depend on
// which IRL entry point happens to expose reconstructed surface polygons.
namespace plicgeom {

// Polygon formed by cutting an axis-aligned box with a plane: its centroid
// and its area. Area is needed as a least-squares weight by the paraboloid
// fit (the `surf` factor in the Fortran paraboloid_fit).
struct PolygonInfo {
  IRL::Pt centroid;
  double area = 0.0;
  std::vector<IRL::Pt> vertices;  // ordered CCW about `normal`; needed by the
                                  // integral fit's boundary integrals
};

inline std::optional<PolygonInfo> planeBoxPolygon(
    const IRL::Pt& lo, const IRL::Pt& hi, const IRL::Normal& normal,
    const double distance) {
  const double vx[2] = {lo[0], hi[0]};
  const double vy[2] = {lo[1], hi[1]};
  const double vz[2] = {lo[2], hi[2]};
  double vpx[8], vpy[8], vpz[8], s[8];
  int n = 0;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c) {
        vpx[n] = vx[a]; vpy[n] = vy[b]; vpz[n] = vz[c];
        s[n] = normal[0] * vpx[n] + normal[1] * vpy[n] + normal[2] * vpz[n] -
               distance;
        ++n;
      }
  // Vertex index bits: (a<<2)|(b<<1)|c, matching the loop nesting above.
  static const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7},
                                   {0, 2}, {1, 3}, {4, 6}, {5, 7},
                                   {0, 4}, {1, 5}, {2, 6}, {3, 7}};

  std::vector<std::array<double, 3>> pts;
  for (const auto& e : edges) {
    const double s0 = s[e[0]], s1 = s[e[1]];
    if ((s0 <= 0.0 && s1 >= 0.0) || (s0 >= 0.0 && s1 <= 0.0)) {
      if (std::abs(s0 - s1) < 1.0e-14) continue;   // edge lies in the plane
      const double t = s0 / (s0 - s1);
      pts.push_back({vpx[e[0]] + t * (vpx[e[1]] - vpx[e[0]]),
                     vpy[e[0]] + t * (vpy[e[1]] - vpy[e[0]]),
                     vpz[e[0]] + t * (vpz[e[1]] - vpz[e[0]])});
    }
  }
  if (pts.size() < 3) return std::nullopt;

  IRL::Normal t0 = IRL::crossProduct(
      normal, std::abs(normal[0]) < 0.9 ? IRL::Normal(1, 0, 0)
                                        : IRL::Normal(0, 1, 0));
  t0.normalize();
  IRL::Normal t1 = IRL::crossProduct(normal, t0);
  t1.normalize();

  double ax = 0.0, ay = 0.0, az = 0.0;
  for (const auto& p : pts) { ax += p[0]; ay += p[1]; az += p[2]; }
  ax /= static_cast<double>(pts.size());
  ay /= static_cast<double>(pts.size());
  az /= static_cast<double>(pts.size());

  std::vector<std::pair<double, int>> order;
  for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
    const IRL::Pt d(pts[i][0] - ax, pts[i][1] - ay, pts[i][2] - az);
    order.emplace_back(std::atan2(t1 * d, t0 * d), i);
  }
  std::sort(order.begin(), order.end());

  double area2 = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const auto& p0 = pts[order[i].second];
    const auto& p1 = pts[order[(i + 1) % order.size()].second];
    const double e1x = p0[0] - ax, e1y = p0[1] - ay, e1z = p0[2] - az;
    const double e2x = p1[0] - ax, e2y = p1[1] - ay, e2z = p1[2] - az;
    const double crx = e1y * e2z - e1z * e2y;
    const double cry = e1z * e2x - e1x * e2z;
    const double crz = e1x * e2y - e1y * e2x;
    const double tri2 = std::sqrt(crx * crx + cry * cry + crz * crz);
    cx += tri2 * (ax + p0[0] + p1[0]) / 3.0;
    cy += tri2 * (ay + p0[1] + p1[1]) / 3.0;
    cz += tri2 * (az + p0[2] + p1[2]) / 3.0;
    area2 += tri2;
  }
  if (area2 < 1.0e-30) return std::nullopt;

  PolygonInfo out;
  out.centroid = IRL::Pt(cx / area2, cy / area2, cz / area2);
  out.area = 0.5 * area2;
  out.vertices.reserve(order.size());
  for (const auto& o : order) {
    const auto& p = pts[o.second];
    out.vertices.emplace_back(p[0], p[1], p[2]);
  }
  return out;
}

// Centroid only -- kept so existing callers (the flatness gate) are unchanged.
inline std::optional<IRL::Pt> planeBoxPolygonCentroid(
    const IRL::Pt& lo, const IRL::Pt& hi, const IRL::Normal& normal,
    const double distance) {
  const auto info = planeBoxPolygon(lo, hi, normal, distance);
  if (!info) return std::nullopt;
  return info->centroid;
}

}  // namespace plicgeom

#endif  // PLIC_GEOMETRY_H_