// Exact stencil moments and centre-cell face labels, all from IRL cuts.
//
// The stencil is N^3 unit cells centred on the origin (N = 3 or 5), cell
// (i,j,k) at (i - (N-1)/2, ...), stored i-major like the rest of the pipeline:
// 7 values per cell = liquid VF, liquid centroid, gas centroid, the centroids
// relative to the cell centre (cell units; 0 for an absent phase).

#ifndef EXAMPLES_R2P_DATAGEN_STENCIL_H_
#define EXAMPLES_R2P_DATAGEN_STENCIL_H_

#include <cmath>
#include <vector>

#include "irl/paraboloid_reconstruction/paraboloid.h"
#include "irl/paraboloid_reconstruction/paraboloid_parametrized_surface.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/generic_cutting/cut_polygon.h"
#include "irl/generic_cutting/generic_cutting.h"
#include "irl/moments/volume_moments.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/r2p_datagen/scene.h"

namespace r2pgen {

inline IRL::Paraboloid toIRL(const Para& p) {
  return IRL::Paraboloid(IRL::Pt(p.d.x(), p.d.y(), p.d.z()),
                         IRL::ReferenceFrame(IRL::Normal(p.e0.x(), p.e0.y(), p.e0.z()),
                                             IRL::Normal(p.e1.x(), p.e1.y(), p.e1.z()),
                                             IRL::Normal(p.e2.x(), p.e2.y(), p.e2.z())),
                         p.a, p.b);
}

inline IRL::PlanarSeparator toIRL(const HalfSpace (&h)[2]) {
  return IRL::PlanarSeparator::fromTwoPlanes(IRL::Plane(IRL::Normal(h[0].n.x(), h[0].n.y(), h[0].n.z()), h[0].c),
                                             IRL::Plane(IRL::Normal(h[1].n.x(), h[1].n.y(), h[1].n.z()), h[1].c),
                                             1.0);
}

inline IRL::RectangularCuboid unitCell(const Vec3& c) {
  return IRL::RectangularCuboid::fromBoundingPts(IRL::Pt(c.x() - 0.5, c.y() - 0.5, c.z() - 0.5),
                                                 IRL::Pt(c.x() + 0.5, c.y() + 0.5, c.z() + 0.5));
}

inline Vec3 cellCentre(int N, int c) {
  const double h = 0.5 * (N - 1);
  return Vec3(c / (N * N) - h, (c / N) % N - h, c % N - h);
}

// A nested sheet is exact only if the lower surface stays below the upper one
// across the stencil. Checked on both surfaces, sampled every `step` cells in
// their own parameters, against the other surface with a small margin.
inline bool nestedInBox(const Para& upper, const Para& lower, double half_width, double step = 0.04,
                        double margin = 1.0e-4) {
  const double box = half_width + 0.05;
  const double reach = std::sqrt(3.0) * box + 0.5;
  auto inBox = [&](const Vec3& x) { return x.cwiseAbs().maxCoeff() <= box; };
  auto check = [&](const Para& surf, const Para& other, double sign) {
    for (double u = -reach; u <= reach; u += step)
      for (double v = -reach; v <= reach; v += step) {
        const Vec3 x = surf.at(u, v);
        if (inBox(x) && sign * other.f(x) < margin) return false;
      }
    return true;
  };
  return check(lower, upper, -1.0) && check(upper, lower, 1.0);
}

// Raw liquid moments (volume, first moment) of one cell.
inline void liquidMoments(const Scene& s, const IRL::RectangularCuboid& cell, double* vol, Vec3* first) {
  auto get = [&](const auto& geom, double sign) {
    const auto m = IRL::getVolumeMoments<IRL::VolumeMoments>(cell, geom);
    *vol += sign * double(m.volume());
    *first += sign * Vec3(m.centroid()[0], m.centroid()[1], m.centroid()[2]);
  };
  *vol = 0.0;
  first->setZero();
  switch (s.kind) {
    case Kind::kSingle: get(toIRL(s.upper), 1.0); break;
    case Kind::kNested: get(toIRL(s.upper), 1.0); get(toIRL(s.lower), -1.0); break;
    case Kind::kPlanes: get(toIRL(s.plane), 1.0); break;
  }
}

// 7 N^3 normalized moments. VF within vf_floor of 0 or 1 is snapped, so the
// round-off of a nested difference never produces a phantom phase.
inline std::vector<double> stencilMoments(const Scene& s, int N, double vf_floor) {
  std::vector<double> out(7 * N * N * N, 0.0);
  for (int c = 0; c < N * N * N; ++c) {
    const Vec3 cc = cellCentre(N, c);
    double v;
    Vec3 m;
    liquidMoments(s, unitCell(cc), &v, &m);
    v = std::min(1.0, std::max(0.0, v));
    double* o = &out[7 * c];
    if (v <= vf_floor) { o[0] = 0.0; continue; }
    if (v >= 1.0 - vf_floor) { o[0] = 1.0; continue; }
    o[0] = v;
    const Vec3 cl = m / v - cc, cg = (cc - m) / (1.0 - v) - cc;
    for (int d = 0; d < 3; ++d) { o[1 + d] = cl[d]; o[4 + d] = cg[d]; }
  }
  return out;
}

struct Faces {
  int n = 0;
  double area[2] = {0.0, 0.0};
  Vec3 normal[2] = {Vec3::Zero(), Vec3::Zero()};   // area-averaged, unit, pointing out of the liquid
};

// The interface inside the centre cell, as up to two faces.
inline Faces centreFaces(const Scene& s) {
  const IRL::RectangularCuboid cell = unitCell(Vec3::Zero());
  Faces f;
  auto surface = [&](const Para& p) {
    return IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(
               cell, toIRL(p))
        .getSurface();
  };
  auto add = [&](int k, double area, const Vec3& n) {
    f.area[k] = area;
    f.normal[k] = n.norm() > 0.0 ? Vec3(n.normalized()) : Vec3::Zero();
  };
  switch (s.kind) {
    case Kind::kSingle: {
      auto surf = surface(s.upper);
      if (s.split_axis < 0) {
        const IRL::Normal n = surf.getAverageNormalNonAligned();
        add(0, surf.getSurfaceArea(), Vec3(n[0], n[1], n[2]));
        break;
      }
      // One surface, two faces: split the triangulation by the sign of the
      // local coordinate across the film (the two sides of a tongue).
      const auto tri = surf.triangulate(0.05);
      Vec3 sum[2] = {Vec3::Zero(), Vec3::Zero()};
      double area[2] = {0.0, 0.0};
      const Vec3 axis = s.split_axis == 0 ? s.upper.e0 : s.upper.e1;
      for (const auto& t : tri.getTriangleList()) {
        const Vec3 p0(t[0][0], t[0][1], t[0][2]), p1(t[1][0], t[1][1], t[1][2]), p2(t[2][0], t[2][1], t[2][2]);
        const double a = 0.5 * (p1 - p0).cross(p2 - p0).norm();
        const Vec3 c = (p0 + p1 + p2) / 3.0;
        const int k = (c - s.upper.d).dot(axis) >= 0.0 ? 1 : 0;
        sum[k] += a * s.upper.outward(c);
        area[k] += a;
      }
      for (int k = 0; k < 2; ++k) add(k, area[k], sum[k]);
      break;
    }
    case Kind::kNested: {
      auto lo = surface(s.lower), up = surface(s.upper);
      const IRL::Normal nl = lo.getAverageNormalNonAligned(), nu = up.getAverageNormalNonAligned();
      add(0, lo.getSurfaceArea(), -Vec3(nl[0], nl[1], nl[2]));   // the liquid is above the lower surface
      add(1, up.getSurfaceArea(), Vec3(nu[0], nu[1], nu[2]));
      break;
    }
    case Kind::kPlanes: {
      const IRL::PlanarSeparator sep = toIRL(s.plane);
      for (int k = 0; k < 2; ++k) {
        const auto m = IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, sep, sep[k]).calculateMoments();
        add(k, std::abs(double(m.volume())), s.plane[k].n);
      }
      break;
    }
  }
  for (int k = 0; k < 2; ++k) if (f.area[k] > 0.0) ++f.n;
  return f;
}

// Where the film's tip (tongue tip curve, the edge's apex line, or the line
// where a thinning sheet's faces would meet) lies:
// 0 inside the centre cell, 1 elsewhere in the 3^3 stencil, 2 outside it,
// -1 if the scene has no tip. Sampled along the curve every 0.01 cells.
inline int tipZone(const Scene& s) {
  std::vector<Vec3> pts;
  const double reach = 15.0;
  if (s.kind == Kind::kSingle && s.split_axis >= 0) {
    for (double t = -reach; t <= reach; t += 0.01)
      pts.push_back(s.split_axis == 1 ? s.upper.at(t, 0.0) : s.upper.at(0.0, t));
  } else if (s.kind == Kind::kPlanes) {
    const Vec3 dir = s.plane[0].n.cross(s.plane[1].n).normalized();
    Mat3 A;
    A.row(0) = s.plane[0].n.transpose();
    A.row(1) = s.plane[1].n.transpose();
    A.row(2) = dir.transpose();
    const Vec3 p0 = A.colPivHouseholderQr().solve(Vec3(s.plane[0].c, s.plane[1].c, 0.0));
    for (double t = -reach; t <= reach; t += 0.01) pts.push_back(p0 + t * dir);
  } else if (s.has_apex) {
    for (double t = -reach; t <= reach; t += 0.01) pts.push_back(s.apex_point + t * s.apex_dir);
  } else {
    return -1;
  }
  double m = 1.0e300;
  for (const auto& p : pts) m = std::min(m, p.cwiseAbs().maxCoeff());
  return m <= 0.5 ? 0 : (m <= 1.5 ? 1 : 2);
}

}  // namespace r2pgen

#endif  // EXAMPLES_R2P_DATAGEN_STENCIL_H_
