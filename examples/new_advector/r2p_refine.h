#ifndef R2P_REFINE_H_
#define R2P_REFINE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "irl/machine_learning_reconstruction/plic_paraboloid.h"
#include "irl/machine_learning_reconstruction/plic_geometry.h"

// R2P extension of plicfit::refineParaboloid.
//
// plicfit fits ONE paraboloid to a neighborhood carrying one plane per cell.
// R2P cells carry two, and the stencil samples two distinct surfaces (the
// two faces of a sheet or film). Fitting one paraboloid to all of it returns
// the mid-surface, whose normal is meaningless for either plane.
//
// This header sorts the stencil's planes into two groups using the R2PNet
// normals as references, then calls plicparab::fitIntegral (or fitPointwise)
// once per group. Everything downstream of the sort is the existing Fortran
// port, unchanged.
//
// NOTE: this is the UNCOUPLED fit, which lets the two normals rotate
// independently and can therefore open a spurious wedge in flat regions.
// See r2p_refine_coupled.h for the coupled version, which is the recommended
// path; this one is retained for A/B comparison.
namespace r2pfit {
 
struct Options {
  int orientation_method = 2;    // 1 = pointwise, 2 = integral (as in plicfit)
  int parab_minpts = 6;          // per GROUP, not per stencil
  double parab_h = 2.5;          // wgauss support radius, in cell widths
  double parab_maxresid = 0.25;  // reject a group's fit above this RMS
  double mesh_size = 1.0;        // pass the cell width; see note below
 
  double max_rotation = 0.35;
  // Cap in radians on departure from the incoming R2PNet normal. A group
  // whose fit exceeds it is clamped, not rejected.
 
  double min_group_area_fraction = 0.10;
  // If either group holds less than this share of total polygon area the
  // split is not credible; keep both network normals.
 
  double min_split_dot = 0.0;
  // A plane must beat this dot product against its assigned group normal to
  // be admitted. 0.0 admits anything on the correct side of perpendicular,
  // matching fitIntegral's own back-face skip.
 
  bool require_both_groups = true;
  // If false, a successful fit on one group is applied even when the other
  // fails, leaving the failed one at its network value.
};
 
// NOTE ON mesh_size. plicfit calls fitIntegral(polys, 1.0, opt.parab_h),
// i.e. mesh_size = 1, which is only correct on a unit-spaced mesh. On the
// advector mesh (dx ~ 1/ncells) every neighbor distance is O(dx), so the
// argument to wgauss is ~0.008 and the kernel returns ~1 for every polygon
// in the stencil: the distance weighting is inert and the residual gate
// never fires. The fitted NORMAL is unaffected (F_t and F_s are
// dimensionless), but curvature comes out in the wrong units and the
// integral fit's normal equations are badly conditioned. Pass the true cell
// width. r2ppass::run does this automatically when the value is left at 1.0.
 
// Cell geometry is carried as lo/hi corner points rather than a
// RectangularCuboid because plicgeom::planeBoxPolygon takes corners, and the
// pass builds them straight from the mesh anyway.
struct CellPlanes {
  IRL::Pt lo;
  IRL::Pt hi;
  IRL::PlanarSeparator separator;
  bool mixed = false;
};
 
struct Result {
  bool fitted[2] = {false, false};
  double residual[2] = {0.0, 0.0};
  double curvature[2] = {0.0, 0.0};
  int count[2] = {0, 0};
  double rotation[2] = {0.0, 0.0};
  double area_fraction[2] = {0.0, 0.0};
  bool split_failed = false;
};
 
namespace detail {
 
// One plane of one cell, tagged with its group.
struct Tagged {
  plicparab::SurfacePolygon poly;
  int group = -1;
  bool is_center = false;
};
 
// Rotates `from` toward `to` by at most `max_angle` radians (Rodrigues).
inline IRL::Normal limitedRotation(const IRL::Normal& from,
                                   const IRL::Normal& to,
                                   const double max_angle, double* achieved) {
  const double raw = from * to;
  const double cos_angle = std::max(-1.0, std::min(1.0, raw));
  const double angle = std::acos(cos_angle);
  const double target = std::min(angle, max_angle);
  *achieved = target;
  if (angle < 1.0e-12 || target < 1.0e-12) return from;
 
  IRL::Normal axis = IRL::crossProduct(from, to);
  if (axis.calculateMagnitude() < 1.0e-12) return from;
  axis.normalize();
 
  const double c = std::cos(target);
  const double s = std::sin(target);
  IRL::Normal out = from * c + IRL::crossProduct(axis, from) * s +
                    axis * (axis * from) * (1.0 - c);
  out.normalize();
  return out;
}
 
// Area and centroid of an ordered, planar polygon in 3D, by fan
// triangulation from vertex 0. Returns false on a degenerate polygon.
inline bool polygonAreaCentroid(const std::vector<IRL::Pt>& v, double* area,
                                IRL::Pt* centroid) {
  const std::size_t n = v.size();
  if (n < 3) return false;
 
  double total = 0.0;
  double cx = 0.0, cy = 0.0, cz = 0.0;
  for (std::size_t i = 1; i + 1 < n; ++i) {
    const IRL::Pt e1(v[i][0] - v[0][0], v[i][1] - v[0][1], v[i][2] - v[0][2]);
    const IRL::Pt e2(v[i + 1][0] - v[0][0], v[i + 1][1] - v[0][1],
                     v[i + 1][2] - v[0][2]);
    const double cross_x = e1[1] * e2[2] - e1[2] * e2[1];
    const double cross_y = e1[2] * e2[0] - e1[0] * e2[2];
    const double cross_z = e1[0] * e2[1] - e1[1] * e2[0];
    const double tri = 0.5 * std::sqrt(cross_x * cross_x + cross_y * cross_y +
                                       cross_z * cross_z);
    if (tri <= 0.0) continue;
    total += tri;
    cx += tri * (v[0][0] + v[i][0] + v[i + 1][0]) / 3.0;
    cy += tri * (v[0][1] + v[i][1] + v[i + 1][1]) / 3.0;
    cz += tri * (v[0][2] + v[i][2] + v[i + 1][2]) / 3.0;
  }
  if (total <= 0.0) return false;
 
  *area = total;
  *centroid = IRL::Pt(cx / total, cy / total, cz / total);
  return true;
}
 
// Sutherland-Hodgman clip of a convex planar polygon to the half-space
// {x : normal . x - distance <= 0}. Vertex order is preserved.
inline std::vector<IRL::Pt> clipToHalfSpace(const std::vector<IRL::Pt>& in,
                                            const IRL::Normal& normal,
                                            const double distance) {
  std::vector<IRL::Pt> out;
  const std::size_t n = in.size();
  if (n == 0) return out;
  out.reserve(n + 2);
 
  for (std::size_t i = 0; i < n; ++i) {
    const IRL::Pt& cur = in[i];
    const IRL::Pt& nxt = in[(i + 1) % n];
    const double dc = normal * cur - distance;
    const double dn = normal * nxt - distance;
 
    if (dc <= 0.0) out.push_back(cur);
    if ((dc < 0.0 && dn > 0.0) || (dc > 0.0 && dn < 0.0)) {
      const double t = dc / (dc - dn);
      out.push_back(IRL::Pt(cur[0] + t * (nxt[0] - cur[0]),
                            cur[1] + t * (nxt[1] - cur[1]),
                            cur[2] + t * (nxt[2] - cur[2])));
    }
  }
  return out;
}
 
// Converts one plane of one cell's separator into a SurfacePolygon.
//
// plicgeom::planeBoxPolygon cuts the plane against the BOX only, which is
// exact for single-plane PLIC but returns the untruncated plane for an R2P
// wedge -- overstating both area and centroid offset on exactly the cells
// where the wedge is tightest. So the box polygon is then clipped against
// every OTHER plane in the separator, leaving only the part that is really
// interface.
//
// The clip half-space is the same regardless of flip state: flipping renames
// the phases but does not move the geometry.
inline std::optional<plicparab::SurfacePolygon> polygonFor(
    const IRL::Pt& lo, const IRL::Pt& hi, const IRL::PlanarSeparator& sep,
    const IRL::UnsignedIndex_t p) {
  const auto info =
      plicgeom::planeBoxPolygon(lo, hi, sep[p].normal(), sep[p].distance());
  if (!info) return std::nullopt;
 
  std::vector<IRL::Pt> verts = info->vertices;
  for (IRL::UnsignedIndex_t q = 0; q < sep.getNumberOfPlanes(); ++q) {
    if (q == p) continue;
    verts = clipToHalfSpace(verts, sep[q].normal(), sep[q].distance());
    if (verts.size() < 3) return std::nullopt;
  }
 
  double area = 0.0;
  IRL::Pt centroid;
  if (!polygonAreaCentroid(verts, &area, &centroid)) return std::nullopt;
  if (area <= 0.0) return std::nullopt;
 
  plicparab::SurfacePolygon sp;
  sp.vertices = verts;
  sp.centroid = centroid;
  sp.normal = sep[p].normal();
  sp.normal.normalize();
  sp.area = area;
  return sp;
}
 
}  // namespace detail
 
// ---------------------------------------------------------------------------
// SORT
// ---------------------------------------------------------------------------
// Assignment is by dot product against the two reference normals, mirroring
// the same-facing / opposite-facing partition that Zonghao's colinearity
// metric already computes in R2P3D_Net (norm_pos accumulates dot >= 0,
// norm_neg accumulates dot < 0). For the classes routed to R2P -- sheet (4)
// and sheet end (6) -- the two normals are near-antiparallel, so this is
// well conditioned.
//
// Position deliberately plays no part. The two faces of a film are under a
// cell apart, at or below the resolution of the data, while their normals
// differ by nearly 180 degrees.
//
// A cell holding two planes has them assigned JOINTLY: of the two possible
// pairings, the one with the greater total dot product wins. Assigning
// independently lets both planes of one neighbor land in the same group,
// which double-counts one surface and starves the other.
inline void sortPlanes(std::vector<detail::Tagged>& tagged,
                       const std::vector<std::size_t>& cell_begin,
                       const IRL::Normal& n0, const IRL::Normal& n1,
                       const Options& opt) {
  for (std::size_t c = 0; c + 1 < cell_begin.size(); ++c) {
    const std::size_t begin = cell_begin[c];
    const std::size_t end = cell_begin[c + 1];
    const std::size_t count = end - begin;
 
    if (count == 1) {
      detail::Tagged& t = tagged[begin];
      const double d0 = t.poly.normal * n0;
      const double d1 = t.poly.normal * n1;
      const int g = (d0 >= d1) ? 0 : 1;
      t.group = (std::max(d0, d1) > opt.min_split_dot) ? g : -1;
    } else if (count >= 2) {
      detail::Tagged& a = tagged[begin];
      detail::Tagged& b = tagged[begin + 1];
      const double a0 = a.poly.normal * n0;
      const double a1 = a.poly.normal * n1;
      const double b0 = b.poly.normal * n0;
      const double b1 = b.poly.normal * n1;
      // Pairing A: a->0, b->1.  Pairing B: a->1, b->0.
      if (a0 + b1 >= a1 + b0) {
        a.group = (a0 > opt.min_split_dot) ? 0 : -1;
        b.group = (b1 > opt.min_split_dot) ? 1 : -1;
      } else {
        a.group = (a1 > opt.min_split_dot) ? 1 : -1;
        b.group = (b0 > opt.min_split_dot) ? 0 : -1;
      }
      // Any further planes in this cell (rare) fall back to the 1-plane rule.
      for (std::size_t extra = begin + 2; extra < end; ++extra) {
        detail::Tagged& t = tagged[extra];
        const double d0 = t.poly.normal * n0;
        const double d1 = t.poly.normal * n1;
        const int g = (d0 >= d1) ? 0 : 1;
        t.group = (std::max(d0, d1) > opt.min_split_dot) ? g : -1;
      }
    }
  }
}
 
// Flattens every plane in the stencil into Tagged polygons, recording where
// each cell's run starts so the sort can pair a cell's two planes. Shared
// with the coupled path.
inline void flattenStencil(const std::vector<CellPlanes>& cells,
                           std::vector<detail::Tagged>* tagged,
                           std::vector<std::size_t>* cell_begin) {
  tagged->clear();
  cell_begin->clear();
  cell_begin->push_back(0);
  for (std::size_t c = 0; c < cells.size(); ++c) {
    if (cells[c].mixed) {
      const IRL::PlanarSeparator& sep = cells[c].separator;
      for (IRL::UnsignedIndex_t p = 0; p < sep.getNumberOfPlanes(); ++p) {
        std::optional<plicparab::SurfacePolygon> sp =
            detail::polygonFor(cells[c].lo, cells[c].hi, sep, p);
        if (!sp) continue;
        detail::Tagged t;
        t.poly = *sp;
        t.is_center = (c == 0);
        tagged->push_back(t);
      }
    }
    cell_begin->push_back(tagged->size());
  }
}
 
// Area share held by each group, after sorting. Returns false if there is no
// area at all.
inline bool groupAreaFractions(const std::vector<detail::Tagged>& tagged,
                               double* fraction) {
  double area[2] = {0.0, 0.0};
  double total = 0.0;
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    if (tagged[i].group >= 0) area[tagged[i].group] += tagged[i].poly.area;
    total += tagged[i].poly.area;
  }
  if (total <= 0.0) return false;
  fraction[0] = area[0] / total;
  fraction[1] = area[1] / total;
  return true;
}
 
// Gathers one group's polygons with the CENTER-CELL polygon first: plicparab
// takes polys[0] as the reference point and frame, so the fitted normal is
// the surface normal at THIS cell's interface rather than a neighbor's.
inline std::vector<plicparab::SurfacePolygon> gatherGroup(
    const std::vector<detail::Tagged>& tagged, const int group) {
  std::vector<plicparab::SurfacePolygon> polys;
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    if (tagged[i].group == group && tagged[i].is_center) {
      polys.push_back(tagged[i].poly);
    }
  }
  if (polys.empty()) return polys;   // no center polygon: caller must bail
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    if (tagged[i].group == group && !tagged[i].is_center) {
      polys.push_back(tagged[i].poly);
    }
  }
  return polys;
}
 
// ---------------------------------------------------------------------------
// ENTRY POINT (uncoupled)
// ---------------------------------------------------------------------------
// cells[0] must be the center cell, and its separator must hold the two
// R2PNet planes in the same order as normal1/normal2. normal1/normal2 must
// already be in final convention (mesh-scaled, normalized, flip-applied).
// They are left unchanged unless this returns true.
inline bool refineTwoNormals(const std::vector<CellPlanes>& cells,
                             IRL::Normal& normal1, IRL::Normal& normal2,
                             Result* result = nullptr,
                             const Options& opt = Options()) {
  Result local;
  Result& res = result ? *result : local;
 
  if (cells.empty() || !cells[0].mixed) return false;
  if (cells[0].separator.getNumberOfPlanes() < 2) return false;
 
  std::vector<detail::Tagged> tagged;
  std::vector<std::size_t> cell_begin;
  flattenStencil(cells, &tagged, &cell_begin);
  if (tagged.size() < 2 * static_cast<std::size_t>(opt.parab_minpts)) {
    return false;
  }
 
  const IRL::Normal network0 = normal1;
  const IRL::Normal network1 = normal2;
  sortPlanes(tagged, cell_begin, network0, network1, opt);
 
  if (!groupAreaFractions(tagged, res.area_fraction)) return false;
  if (std::min(res.area_fraction[0], res.area_fraction[1]) <
      opt.min_group_area_fraction) {
    res.split_failed = true;
    return false;
  }
 
  IRL::Normal fitted_normal[2] = {network0, network1};
  bool any = false;
 
  for (int g = 0; g < 2; ++g) {
    const IRL::Normal& network = (g == 0) ? network0 : network1;
    const std::vector<plicparab::SurfacePolygon> polys = gatherGroup(tagged, g);
    res.count[g] = static_cast<int>(polys.size());
    if (polys.empty()) continue;                 // no center polygon
    if (res.count[g] < opt.parab_minpts) continue;
 
    const std::optional<plicparab::FitResult> fit =
        (opt.orientation_method == 2)
            ? plicparab::fitIntegral(polys, opt.mesh_size, opt.parab_h)
            : plicparab::fitPointwise(polys, opt.mesh_size, opt.parab_h);
    if (!fit) continue;
    if (fit->rms_residual > opt.parab_maxresid) continue;
 
    IRL::Normal candidate = fit->normal;
    if (candidate * network < 0.0) candidate = -candidate;
 
    fitted_normal[g] = detail::limitedRotation(network, candidate,
                                               opt.max_rotation,
                                               &res.rotation[g]);
    res.fitted[g] = true;
    res.residual[g] = fit->rms_residual;
    res.curvature[g] = fit->curvature;
    any = true;
  }
 
  if (opt.require_both_groups && !(res.fitted[0] && res.fitted[1])) return false;
  if (!any) return false;
 
  normal1 = fitted_normal[0];
  normal2 = fitted_normal[1];
  return true;
}
 
}  // namespace r2pfit
 
#endif  // R2P_REFINE_H_