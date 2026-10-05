// Sample families and their placement on the grid.
//
// Every family is built in a canonical frame -- film normal +z, the direction
// the film changes along (thickness gradient, tip) +x -- from exactly the
// geometry IRL can cut exactly:
//
//   kSingle  liquid = below one paraboloid                  (bulk, tongue)
//   kNested  liquid = below(upper) - below(lower), with the  (sheet, neck)
//            lower surface verified to lie under the upper
//            one throughout the stencil, so the difference
//            of two exact cuts is exact
//   kPlanes  liquid = below both of two planes               (edge)
//
// Films are always liquid: which phase the network sees is decided by the
// deployment's phase-0 rule, which is symmetric under swapping the phases, so
// a gas film would reproduce the same inputs and labels.
//
// Each family also supplies an anchor: a point on its surface, with the
// outward normal there. Placement rotates the scene uniformly (optionally
// near an axis-aligned orientation) and puts the anchor's tangent plane
// through the centre cell, so the centre cell is always cut.
//
// Paraboloid convention (IRL): in the local frame (e0, e1, e2) about datum d
// the surface is h = -(a u^2 + b v^2), and the liquid is below it, i.e. where
// f(x) = h + a u^2 + b v^2 < 0 with (u, v, h) the local coordinates of x.

#ifndef EXAMPLES_R2P_DATAGEN_SCENE_H_
#define EXAMPLES_R2P_DATAGEN_SCENE_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>

#include "examples/r2p_datagen/config.h"

namespace r2pgen {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

constexpr double kPi = 3.14159265358979323846;
inline double deg2rad(double d) { return d * kPi / 180.0; }

struct Para {
  Vec3 d = Vec3::Zero(), e0 = Vec3::UnitX(), e1 = Vec3::UnitY(), e2 = Vec3::UnitZ();
  double a = 0.0, b = 0.0;

  double f(const Vec3& x) const {
    const Vec3 r = x - d;
    const double u = r.dot(e0), v = r.dot(e1);
    return r.dot(e2) + a * u * u + b * v * v;
  }
  Vec3 at(double u, double v) const { return d + u * e0 + v * e1 - (a * u * u + b * v * v) * e2; }
  Vec3 outward(const Vec3& x) const {   // unit normal pointing away from the liquid
    const Vec3 r = x - d;
    return (e2 + 2.0 * a * r.dot(e0) * e0 + 2.0 * b * r.dot(e1) * e1).normalized();
  }
};

struct HalfSpace {   // liquid side: n.x <= c
  Vec3 n = Vec3::UnitZ();
  double c = 0.0;
};

enum class Kind { kSingle, kNested, kPlanes };

struct Truth {
  int family = -1;
  double thickness = -1.0;     // film thickness at the anchor (cells), -1 if not a film
  double a = 0.0, b = 0.0;     // paraboloid coefficients (sheet: mid-surface; bulk, tongue: the surface)
  double splay = 0.0;          // opening angle between the faces at the anchor (deg)
  double flare = 0.0;          // thickness curvature (1/cells)
  double tip_radius = -1.0, tip_distance = -1.0;
  double neck_thickness = -1.0, neck_distance = -1.0;   // neck: thinnest point, and the anchor's distance to it (cells)
};

struct Scene {
  Kind kind = Kind::kSingle;
  Para upper, lower;
  HalfSpace plane[2];
  int split_axis = -1;   // kSingle only: split the surface into two faces by the sign of local u (0) or v (1)
  Vec3 anchor = Vec3::Zero(), anchor_normal = Vec3::UnitZ();
  bool has_apex = false;   // sheet with splay: the line where its (flat) faces would meet
  Vec3 apex_point = Vec3::Zero(), apex_dir = Vec3::UnitY();
  Truth truth;

  // Point membership, for the self-test.
  bool liquid(const Vec3& x) const {
    switch (kind) {
      case Kind::kSingle: return upper.f(x) < 0.0;
      case Kind::kNested: return upper.f(x) < 0.0 && lower.f(x) >= 0.0;
      case Kind::kPlanes: return plane[0].n.dot(x) <= plane[0].c && plane[1].n.dot(x) <= plane[1].c;
    }
    return false;
  }
};

enum class FamilyKind { kSheet, kTongue, kEdge, kBulk, kNeck };

// One [section] of the config: a family kind plus the distributions of its
// parameters. Only the keys of its own kind are read.
struct Family {
  std::string name;
  FamilyKind kind = FamilyKind::kSheet;
  double weight = 1.0;
  Dist a, b;                          // sheet, bulk, tongue
  Dist thickness, splay, flare;       // sheet
  Dist apex_distance;                 // sheet, alternative to splay
  bool use_apex = false;
  Dist tip_distance;                  // tongue, edge
  Dist opening;                       // edge, neck (at the anchor)
  Dist neck_thickness, vertex_distance, flare_split;   // neck

  static Family fromSection(const Section& s) {
    Family f;
    f.name = s.name;
    const std::string k = s.str("kind", "");
    if (k == "sheet") f.kind = FamilyKind::kSheet;
    else if (k == "tongue") f.kind = FamilyKind::kTongue;
    else if (k == "edge") f.kind = FamilyKind::kEdge;
    else if (k == "bulk") f.kind = FamilyKind::kBulk;
    else if (k == "neck") f.kind = FamilyKind::kNeck;
    else throw std::runtime_error("[" + s.name + "] kind must be sheet, tongue, edge, bulk or neck");
    f.weight = s.num("weight", 1.0);
    if (f.kind != FamilyKind::kEdge) {
      f.a = s.dist("a", "0");
      f.b = s.dist("b", f.kind == FamilyKind::kTongue ? "loguniform 0.25 10" : "0");
    }
    if (f.kind == FamilyKind::kSheet) {
      f.thickness = s.dist("thickness", "loguniform 0.005 1");
      if (s.has("apex_distance")) {
        if (s.has("splay")) throw std::runtime_error("[" + s.name + "] give splay or apex_distance, not both");
        f.use_apex = true;
        f.apex_distance = s.dist("apex_distance", "");
      } else {
        f.splay = s.dist("splay", "0");
      }
      f.flare = s.dist("flare", "0");
    }
    if (f.kind == FamilyKind::kTongue || f.kind == FamilyKind::kEdge)
      f.tip_distance = s.dist("tip_distance", "uniform 0 3");
    if (f.kind == FamilyKind::kEdge) f.opening = s.dist("opening", "uniform 5 120");
    if (f.kind == FamilyKind::kNeck) {
      f.neck_thickness = s.dist("neck_thickness", "loguniform 0.005 0.3");
      f.vertex_distance = s.dist("vertex_distance", "uniform 0.5 2.5");
      f.opening = s.dist("opening", "uniform 5 40");
      f.flare_split = s.dist("flare_split", "uniform 0 1");
    }
    return f;
  }

  // Builds the scene in the canonical frame. False rejects this parameter
  // draw (e.g. a sheet thicker than its own radius of curvature).
  //
  // a, b are IRL paraboloid coefficients, surface h = -(a u^2 + b v^2) along
  // the local normal (liquid below): for a sheet those of the mid-surface
  // (film normal = +z), for bulk and tongue those of the surface itself.
  template <class Engine>
  bool build(Engine& eng, Scene* sc) const {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const bool top = u01(eng) < 0.5;   // which face carries the anchor
    *sc = Scene();
    Truth& tr = sc->truth;

    switch (kind) {
      case FamilyKind::kSheet: {
        // Faces at +-t/2 about the mid-surface z = -(a x^2 + b y^2) (principal
        // curvatures k = -2a, -2b), each with the curvature of its offset
        // surface, tilted by +-splay/2 about y (thickness grows along +x) and
        // bent apart by the flare.
        const double t = thickness(eng);
        const double ca = a(eng), cb = b(eng);
        const double k1 = -2.0 * ca, k2 = -2.0 * cb;
        // A thinning sheet is given either its opening angle directly (splay)
        // or the distance to the apex where its faces would meet, which fixes
        // the angle for this thickness: splay = 2 atan(t / (2 d)).
        double th = 0.0;
        if (use_apex) {
          const double d = apex_distance(eng);
          if (!(d > 0.0)) return false;
          th = 2.0 * std::atan(t / (2.0 * d));
        } else {
          th = deg2rad(splay(eng));
        }
        const double fl = flare(eng);
        const double h = 0.5 * t;
        if (!(t > 0.0) || std::abs(k1) * h >= 0.9 || std::abs(k2) * h >= 0.9) return false;
        const double c = std::cos(0.5 * th), s = std::sin(0.5 * th);
        Para& up = sc->upper;
        up.d = Vec3(0, 0, h);
        up.e0 = Vec3(c, 0, s);
        up.e1 = Vec3::UnitY();
        up.e2 = Vec3(-s, 0, c);
        up.a = -(0.5 * k1 / (1.0 - k1 * h) + 0.5 * fl);
        up.b = -0.5 * k2 / (1.0 - k2 * h);
        Para& lo = sc->lower;
        lo.d = Vec3(0, 0, -h);
        lo.e0 = Vec3(c, 0, -s);
        lo.e1 = Vec3::UnitY();
        lo.e2 = Vec3(s, 0, c);
        lo.a = -(0.5 * k1 / (1.0 + k1 * h) - 0.5 * fl);
        lo.b = -0.5 * k2 / (1.0 + k2 * h);
        sc->kind = Kind::kNested;
        sc->anchor = top ? up.d : lo.d;
        sc->anchor_normal = top ? up.e2 : Vec3(-lo.e2);
        tr.thickness = t; tr.a = ca; tr.b = cb; tr.splay = th * 180.0 / kPi; tr.flare = fl;
        if (th > 0.0) {
          sc->has_apex = true;
          tr.tip_distance = 0.5 * t / std::tan(0.5 * th);
          sc->apex_point = Vec3(-tr.tip_distance, 0, 0);
        }
        return true;
      }
      case FamilyKind::kNeck: {
        // A wedge that levels off into a thin film instead of meeting at a
        // tip: two faces z = +-t0/2 at the vertex x = 0 (the thinnest point),
        // both with the shared bending (a, b) of the mid-surface (as a sheet),
        // curving apart along x so the gap grows as t0 + f x^2 -- the upper
        // face takes f*w of that, the lower f*(1-w) (w = 0 or 1: one face
        // flat). The anchor sits at x = d on one face, where the faces open by
        // ~opening: f = tan(opening/2) / d (exact for w = 1/2 and a = 0). With
        // f > 0 and t0 > 0 the faces cannot cross.
        const double t0 = neck_thickness(eng), d = vertex_distance(eng);
        const double th = deg2rad(opening(eng)), w = flare_split(eng);
        const double ca = a(eng), cb = b(eng);
        const double k1 = -2.0 * ca, k2 = -2.0 * cb, h = 0.5 * t0;
        if (!(t0 > 0.0) || !(d > 0.0) || !(th > 0.0 && th < kPi) || w < 0.0 || w > 1.0) return false;
        if (std::abs(k1) * h >= 0.9 || std::abs(k2) * h >= 0.9) return false;
        const double f = std::tan(0.5 * th) / d;
        Para& up = sc->upper;
        up.d = Vec3(0, 0, h);
        up.a = -(0.5 * k1 / (1.0 - k1 * h) + f * w);
        up.b = -0.5 * k2 / (1.0 - k2 * h);
        Para& lo = sc->lower;
        lo.d = Vec3(0, 0, -h);
        lo.a = -(0.5 * k1 / (1.0 + k1 * h) - f * (1.0 - w));
        lo.b = -0.5 * k2 / (1.0 + k2 * h);
        sc->kind = Kind::kNested;
        const Vec3 pu = up.at(d, 0.0), pl = lo.at(d, 0.0);
        const Vec3 nu = up.outward(pu), nl = -lo.outward(pl);   // outward from the film
        sc->anchor = top ? pu : pl;
        sc->anchor_normal = top ? nu : nl;
        tr.thickness = pu.z() - pl.z();
        tr.a = ca; tr.b = cb;
        tr.splay = std::acos(std::max(-1.0, std::min(1.0, -nu.dot(nl)))) * 180.0 / kPi;
        tr.flare = f;
        tr.neck_thickness = t0; tr.neck_distance = d;
        return true;
      }
      case FamilyKind::kTongue: {
        // Parabolic tongue x = -(a y^2 + b z^2): a film ending in a rounded
        // tip of radius 1/(2b), thickness 2 sqrt(s/b) at distance s behind it;
        // a bends the edge line (a > 0: convex, like a disc rim).
        const double ca = a(eng), cb = b(eng), s = tip_distance(eng);
        if (!(cb > 0.0) || s < 0.0) return false;
        Para& p = sc->upper;
        p.d = Vec3::Zero();
        p.e0 = Vec3::UnitY();
        p.e1 = Vec3::UnitZ();
        p.e2 = Vec3::UnitX();
        p.a = ca;
        p.b = cb;
        const double z = std::sqrt(s / p.b) * (top ? 1.0 : -1.0);
        sc->kind = Kind::kSingle;
        sc->split_axis = 1;
        sc->anchor = Vec3(-s, 0, z);
        sc->anchor_normal = p.outward(sc->anchor);
        tr.thickness = 2.0 * std::sqrt(s / cb); tr.tip_radius = 0.5 / cb; tr.tip_distance = s;
        tr.a = ca; tr.b = cb;
        return true;
      }
      case FamilyKind::kEdge: {
        // Two planes meeting on the y axis, opening toward -x.
        const double phi = deg2rad(opening(eng)), s = tip_distance(eng);
        if (!(phi > 0.0 && phi < kPi) || s < 0.0) return false;
        const double c = std::cos(0.5 * phi), sn = std::sin(0.5 * phi);
        sc->plane[0].n = Vec3(sn, 0, c);
        sc->plane[1].n = Vec3(sn, 0, -c);
        sc->kind = Kind::kPlanes;
        sc->anchor = s * Vec3(-c, 0, top ? sn : -sn);
        sc->anchor_normal = sc->plane[top ? 0 : 1].n;
        tr.thickness = 2.0 * s * std::tan(0.5 * phi); tr.splay = phi * 180.0 / kPi; tr.tip_distance = s;
        return true;
      }
      case FamilyKind::kBulk: {
        Para& p = sc->upper;
        p.a = a(eng);
        p.b = b(eng);
        sc->kind = Kind::kSingle;
        sc->anchor = Vec3::Zero();
        sc->anchor_normal = Vec3::UnitZ();
        tr.a = p.a; tr.b = p.b;
        return true;
      }
    }
    return false;
  }
};

struct Placement {
  double band_fraction = 0.5;      // share placed uniformly across the band where the anchor plane cuts the cell
  double aligned_fraction = 0.0;   // share rotated near a grid-aligned orientation
  double jitter_deg = 10.0;        // max deviation from that orientation

  template <class Engine>
  static Mat3 randomRotation(Engine& eng) {
    std::normal_distribution<double> n01(0.0, 1.0);
    Eigen::Quaterniond q(n01(eng), n01(eng), n01(eng), n01(eng));
    q.normalize();
    return q.toRotationMatrix();
  }

  template <class Engine>
  Mat3 rotation(Engine& eng) const {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(eng) >= aligned_fraction) return randomRotation(eng);
    // Random signed axis permutation (det +1), then a small random rotation.
    int perm[3] = {0, 1, 2};
    std::shuffle(perm, perm + 3, eng);
    Mat3 P = Mat3::Zero();
    for (int r = 0; r < 3; ++r) P(r, perm[r]) = u01(eng) < 0.5 ? -1.0 : 1.0;
    if (P.determinant() < 0.0) P.row(0) *= -1.0;
    Vec3 axis(std::normal_distribution<double>(0, 1)(eng), std::normal_distribution<double>(0, 1)(eng),
              std::normal_distribution<double>(0, 1)(eng));
    const double angle = deg2rad(jitter_deg) * u01(eng);
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix() * P;
  }

  // Maps a canonical scene onto the grid: x_grid = Q (x - anchor) + o.
  template <class Engine>
  Scene place(const Scene& in, Engine& eng) const {
    std::uniform_real_distribution<double> u(-0.5, 0.5), u01(0.0, 1.0);
    const Mat3 Q = rotation(eng);
    const Vec3 n = Q * in.anchor_normal;
    Vec3 o(u(eng), u(eng), u(eng));   // anchor uniformly inside the centre cell
    if (u01(eng) < band_fraction) {
      // Anchor tangent plane offset uniformly over the whole band where it
      // cuts the cell: more slivers than the volume placement above.
      const double e = 0.5 * n.cwiseAbs().sum();
      o += (u(eng) * 2.0 * e - o.dot(n)) * n;
    }
    auto movePara = [&](const Para& p) {
      Para q = p;
      q.d = Q * (p.d - in.anchor) + o;
      q.e0 = Q * p.e0; q.e1 = Q * p.e1; q.e2 = Q * p.e2;
      return q;
    };
    Scene out = in;
    out.upper = movePara(in.upper);
    out.lower = movePara(in.lower);
    for (int k = 0; k < 2; ++k) {
      out.plane[k].n = Q * in.plane[k].n;
      out.plane[k].c = in.plane[k].c - in.plane[k].n.dot(in.anchor) + out.plane[k].n.dot(o);
    }
    out.anchor = o;
    out.anchor_normal = n;
    out.apex_point = Q * (in.apex_point - in.anchor) + o;
    out.apex_dir = Q * in.apex_dir;
    return out;
  }
};

}  // namespace r2pgen

#endif  // EXAMPLES_R2P_DATAGEN_SCENE_H_
