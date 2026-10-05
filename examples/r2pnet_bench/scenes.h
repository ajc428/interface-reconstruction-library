// Benchmark scene generators.
//
// Every scene is built in a local frame where the film's mid-surface passes
// through the origin with normal +z, then randomly rotated and translated so
// an anchor point on the interface lands uniformly inside the centre cell
// (the cell [-0.5,0.5]^3, mesh spacing 1). That guarantees the centre cell is
// cut, while the geometry around it is whatever the category dictates.
//
// Thicknesses and radii are in cell widths.

#ifndef EXAMPLES_R2PNET_BENCH_SCENES_H_
#define EXAMPLES_R2PNET_BENCH_SCENES_H_

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "examples/r2pnet_bench/geometry.h"

namespace bench {

struct Scene {
  std::string category;
  Region region;
  double thickness = 0.0;   // film thickness at the anchor (0 = not a film)
  double radius1 = 0.0;     // principal radii of the mid-surface (0 = flat)
  double radius2 = 0.0;
  double param = 0.0;       // category-specific (wedge slope, rim bulb ratio, ...)
};

inline const std::vector<std::string>& allCategories() {
  static const std::vector<std::string> cats = {
      "flat_thin",   "flat_thick", "sphere_shell_thin", "sphere_shell_thick",
      "cyl_shell_thin", "cyl_shell_thick", "saddle_thin", "wedge",
      "rim",         "junction",   "bulk_sphere"};
  return cats;
}

class SceneSampler {
 public:
  explicit SceneSampler(unsigned long long seed) : eng_(seed) {}

  Scene sample(const std::string& cat) {
    Scene s;
    s.category = cat;
    Vec3 anchor{};
    if (cat == "flat_thin" || cat == "flat_thick") {
      s.thickness = cat == "flat_thin" ? logUniform(0.005, 0.5) : logUniform(0.5, 3.0);
      s.region.root = buildSlab(&s.region, s.thickness);
      anchor = {0.0, 0.0, faceSign() * 0.5 * s.thickness};
    } else if (cat == "sphere_shell_thin" || cat == "sphere_shell_thick" ||
               cat == "cyl_shell_thin" || cat == "cyl_shell_thick") {
      const bool thin = cat.find("thin") != std::string::npos;
      const bool sphere = cat.find("sphere") != std::string::npos;
      s.thickness = thin ? logUniform(0.005, 0.5) : logUniform(0.5, 3.0);
      // Mid-surface radius of curvature. Deformation3D sheets sit at roughly
      // 5-20 cells; the range brackets that on both sides.
      const double R = std::max(logUniform(2.0, 40.0), 1.5 * s.thickness);
      s.radius1 = R;
      s.radius2 = sphere ? R : 0.0;
      const Vec3 centre{0.0, 0.0, -R};
      const double ro = R + 0.5 * s.thickness, ri = R - 0.5 * s.thickness;
      Region& g = s.region;
      if (sphere) {
        g.root = g.opAnd(g.prim(Quadric::sphere(centre, ro)),
                         g.opNot(g.prim(Quadric::sphere(centre, ri))));
      } else {
        const Vec3 axis{1.0, 0.0, 0.0};
        g.root = g.opAnd(g.prim(Quadric::cylinder(centre, axis, ro)),
                         g.opNot(g.prim(Quadric::cylinder(centre, axis, ri))));
      }
      anchor = {0.0, 0.0, faceSign() * 0.5 * s.thickness};
    } else if (cat == "saddle_thin") {
      s.thickness = logUniform(0.005, 0.5);
      s.radius1 = logUniform(3.0, 40.0);
      s.radius2 = -logUniform(3.0, 40.0);
      const double alpha = 0.5 / s.radius1, beta = 0.5 / s.radius2;
      Region& g = s.region;
      g.root = g.opAnd(g.prim(Quadric::paraboloidBelow(alpha, beta, 0.5 * s.thickness)),
                       g.opNot(g.prim(Quadric::paraboloidBelow(alpha, beta, -0.5 * s.thickness))));
      anchor = {0.0, 0.0, faceSign() * 0.5 * s.thickness};
    } else if (cat == "wedge") {
      // A thinning sheet: faces tilted +/- theta/2 about the x-axis, so the
      // thickness varies linearly in y with slope 2 tan(theta/2). The line
      // where they would meet is kept outside the stencil (that case is
      // "rim"-like and belongs to generate_sheet_edge territory).
      double t, slope;
      do {
        t = logUniform(0.02, 1.5);
        slope = logUniform(0.01, 0.4);
      } while (t / slope < 6.0);
      s.thickness = t;
      s.param = slope;
      const double half = std::atan(0.5 * slope);
      const Vec3 n_top{0.0, -std::sin(half), std::cos(half)};
      const Vec3 n_bot{0.0, -std::sin(half), -std::cos(half)};
      Region& g = s.region;
      g.root = g.opAnd(g.prim(Quadric::plane(n_top, 0.5 * t * std::cos(half))),
                       g.prim(Quadric::plane(n_bot, 0.5 * t * std::cos(half))));
      anchor = {0.0, 0.0, faceSign() * 0.5 * t};
    } else if (cat == "rim") {
      // Sheet that ends at y = 0. Half the time the end is cut square, half
      // the time it carries a cylindrical bulb (the retracting rim).
      s.thickness = logUniform(0.005, 1.0);
      Region& g = s.region;
      const int slab = buildSlab(&g, s.thickness);
      const int cut = g.prim(Quadric::plane({0.0, 1.0, 0.0}, 0.0));
      int body = g.opAnd(slab, cut);
      if (uniform(0.0, 1.0) < 0.5) {
        const double rb = 0.5 * s.thickness * uniform(1.0, 4.0);
        s.param = rb / (0.5 * s.thickness);
        body = g.opOr(body, g.prim(Quadric::cylinder({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, rb)));
      }
      g.root = body;
      anchor = {0.0, uniform(-2.0, 0.0), faceSign() * 0.5 * s.thickness};
    } else if (cat == "junction") {
      // Sheet running into a bulk body: a large sphere or a pool half-space.
      s.thickness = logUniform(0.005, 1.0);
      Region& g = s.region;
      const int slab = buildSlab(&g, s.thickness);
      double y_surface;
      int blob;
      if (uniform(0.0, 1.0) < 0.5) {
        const double Rb = logUniform(2.0, 10.0);
        s.radius1 = Rb;
        blob = g.prim(Quadric::sphere({0.0, Rb, 0.0}, Rb));
        y_surface = Rb - std::sqrt(std::max(0.0, Rb * Rb - 0.25 * s.thickness * s.thickness));
      } else {
        blob = g.prim(Quadric::plane({0.0, -1.0, 0.0}, 0.0));
        y_surface = 0.0;
      }
      g.root = g.opOr(g.opAnd(slab, g.prim(Quadric::plane({0.0, 1.0, 0.0}, y_surface + 1.0))), blob);
      anchor = {0.0, y_surface + uniform(-2.0, 0.5), faceSign() * 0.5 * s.thickness};
    } else if (cat == "bulk_sphere") {
      // Control case: a single well-resolved surface.
      const double R = logUniform(2.0, 40.0);
      s.radius1 = s.radius2 = R;
      Region& g = s.region;
      g.root = g.prim(Quadric::sphere({0.0, 0.0, -R}, R));
      anchor = {0.0, 0.0, 0.0};
    } else {
      throw std::runtime_error("unknown category " + cat);
    }

    if (s.region.root < 0) throw std::runtime_error("scene " + cat + " has no root node");
    const Mat3 R = randomRotation();
    const Vec3 shift{uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)};
    s.region.transform(R, shift, anchor);
    return s;
  }

  double uniform(double a, double b) { return std::uniform_real_distribution<double>(a, b)(eng_); }
  double normal01() { return std::normal_distribution<double>(0.0, 1.0)(eng_); }
  std::mt19937_64& engine() { return eng_; }

 private:
  double logUniform(double a, double b) { return std::exp(uniform(std::log(a), std::log(b))); }
  double faceSign() { return uniform(0.0, 1.0) < 0.5 ? -1.0 : 1.0; }

  // Slab |z| < t/2 in the local frame; returns its node.
  static int buildSlab(Region* g, double t) {
    return g->opAnd(g->prim(Quadric::plane({0.0, 0.0, 1.0}, 0.5 * t)),
                    g->prim(Quadric::plane({0.0, 0.0, -1.0}, 0.5 * t)));
  }

  // Uniform random rotation (normalized Gaussian quaternion).
  Mat3 randomRotation() {
    double q[4];
    double n = 0.0;
    for (double& x : q) { x = normal01(); n += x * x; }
    n = std::sqrt(n);
    for (double& x : q) x /= n;
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    Mat3 R;
    R[0] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)};
    R[1] = {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)};
    R[2] = {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
    return R;
  }

  std::mt19937_64 eng_;
};

}  // namespace bench

#endif  // EXAMPLES_R2PNET_BENCH_SCENES_H_
