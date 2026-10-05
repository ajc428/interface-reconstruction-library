// Exact(-ish) ground-truth geometry for the R2P-Net benchmark.
//
// A liquid region is a small CSG tree over quadric half-spaces
//
//   q(x) = x.A x + b.x + c < 0          (inside)
//
// which covers planes, spheres, cylinders and paraboloids -- every surface
// the film scenes need, including exact offset surfaces for curved shells.
//
// Moments are integrated column by column: along a line parallel to one axis
// each quadric is a quadratic in the line parameter, so the region's extent on
// that line is a union of intervals found in closed form. Integration is exact
// along the column and midpoint-rule across columns. The column axis is chosen
// per cell as the one best aligned with the nearest surface normal, so thin
// films are always crossed rather than grazed.
//
// Every interval endpoint remembers which primitive produced it, which gives
// the true surface patches (area, mean outward normal) in a cell for free.

#ifndef EXAMPLES_R2PNET_BENCH_GEOMETRY_H_
#define EXAMPLES_R2PNET_BENCH_GEOMETRY_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace bench {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 operator*(double s, const Vec3& a) { return {s * a[0], s * a[1], s * a[2]}; }
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalized(const Vec3& a) {
  const double n = norm(a);
  return n > 0.0 ? (1.0 / n) * a : Vec3{0.0, 0.0, 0.0};
}
inline Vec3 matvec(const Mat3& m, const Vec3& v) {
  return {dot(m[0], v), dot(m[1], v), dot(m[2], v)};
}
inline Mat3 transpose(const Mat3& m) {
  Mat3 t;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) t[i][j] = m[j][i];
  return t;
}
inline Mat3 matmul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) r[i][j] += a[i][k] * b[k][j];
  return r;
}

// ---------------------------------------------------------------------------
// Quadric half-space.
// ---------------------------------------------------------------------------
struct Quadric {
  Mat3 A{};   // symmetric
  Vec3 b{};
  double c = 0.0;

  double eval(const Vec3& x) const { return dot(x, matvec(A, x)) + dot(b, x) + c; }
  Vec3 grad(const Vec3& x) const {
    const Vec3 ax = matvec(A, x);
    return {2.0 * ax[0] + b[0], 2.0 * ax[1] + b[1], 2.0 * ax[2] + b[2]};
  }
  double frobA() const {
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) s += A[i][j] * A[i][j];
    return std::sqrt(s);
  }

  // Express in global coordinates, given x_local = R^T (x - shift) + anchor.
  Quadric transformed(const Mat3& R, const Vec3& shift, const Vec3& anchor) const {
    const Mat3 Rt = transpose(R);
    const Vec3 k = anchor - matvec(Rt, shift);
    Quadric g;
    g.A = matmul(matmul(R, A), Rt);
    const Vec3 Ak = matvec(A, k);
    g.b = matvec(R, Vec3{2.0 * Ak[0] + b[0], 2.0 * Ak[1] + b[1], 2.0 * Ak[2] + b[2]});
    g.c = dot(k, Ak) + dot(b, k) + c;
    return g;
  }

  // Inside = { n.x - d < 0 }
  static Quadric plane(const Vec3& n, double d) {
    Quadric q;
    q.b = normalized(n);
    q.c = -d;
    return q;
  }
  // Inside = ball of radius r about c0
  static Quadric sphere(const Vec3& c0, double r) {
    Quadric q;
    for (int i = 0; i < 3; ++i) q.A[i][i] = 1.0;
    q.b = -2.0 * c0;
    q.c = dot(c0, c0) - r * r;
    return q;
  }
  // Inside = infinite cylinder of radius r, axis direction a through p
  static Quadric cylinder(const Vec3& p, const Vec3& a_in, double r) {
    const Vec3 a = normalized(a_in);
    Quadric q;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) q.A[i][j] = (i == j ? 1.0 : 0.0) - a[i] * a[j];
    const Vec3 Ap = matvec(q.A, p);
    q.b = -2.0 * Ap;
    q.c = dot(p, Ap) - r * r;
    return q;
  }
  // Inside = { z - (alpha x^2 + beta y^2) - z0 < 0 }  (below a paraboloid)
  static Quadric paraboloidBelow(double alpha, double beta, double z0) {
    Quadric q;
    q.A[0][0] = -alpha;
    q.A[1][1] = -beta;
    q.b = {0.0, 0.0, 1.0};
    q.c = -z0;
    return q;
  }
  Quadric negated() const {
    Quadric q = *this;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) q.A[i][j] = -q.A[i][j];
    q.b = -1.0 * q.b;
    q.c = -q.c;
    return q;
  }
};

// ---------------------------------------------------------------------------
// Intervals with provenance.
// ---------------------------------------------------------------------------
struct Interval {
  double lo, hi;
  int tag_lo, tag_hi;   // primitive id that produced the endpoint, -1 = column end
};
using IntervalList = std::vector<Interval>;

inline void complementList(const IntervalList& in, double t0, double t1, IntervalList* out) {
  out->clear();
  double cur = t0;
  int cur_tag = -1;
  for (const auto& iv : in) {
    if (iv.lo > cur) out->push_back({cur, iv.lo, cur_tag, iv.tag_lo});
    cur = iv.hi;
    cur_tag = iv.tag_hi;
  }
  if (t1 > cur) out->push_back({cur, t1, cur_tag, -1});
}

inline void intersectLists(const IntervalList& a, const IntervalList& b, IntervalList* out) {
  out->clear();
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    const double lo = std::max(a[i].lo, b[j].lo);
    const double hi = std::min(a[i].hi, b[j].hi);
    if (hi > lo) {
      out->push_back({lo, hi, a[i].lo >= b[j].lo ? a[i].tag_lo : b[j].tag_lo,
                      a[i].hi <= b[j].hi ? a[i].tag_hi : b[j].tag_hi});
    }
    if (a[i].hi < b[j].hi) ++i; else ++j;
  }
}

inline void unionLists(const IntervalList& a, const IntervalList& b, IntervalList* out) {
  out->clear();
  IntervalList all;
  all.reserve(a.size() + b.size());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(all),
             [](const Interval& x, const Interval& y) { return x.lo < y.lo; });
  for (const auto& iv : all) {
    if (!out->empty() && iv.lo <= out->back().hi) {
      if (iv.hi > out->back().hi) {
        out->back().hi = iv.hi;
        out->back().tag_hi = iv.tag_hi;
      }
    } else {
      out->push_back(iv);
    }
  }
}

// { t in [t0,t1] : q(base + t e_axis) < 0 }
inline void quadricIntervals(const Quadric& q, int tag, const Vec3& base, int axis,
                             double t0, double t1, IntervalList* out) {
  out->clear();
  const double A = q.A[axis][axis];
  const Vec3 Ab = matvec(q.A, base);
  const double B = 2.0 * Ab[axis] + q.b[axis];
  const double C = q.eval(base);
  auto push = [&](double lo, double hi, int tl, int th) {
    if (lo < t0) { lo = t0; tl = -1; }
    if (hi > t1) { hi = t1; th = -1; }
    if (hi > lo) out->push_back({lo, hi, tl, th});
  };
  const double tiny = 1.0e-14;
  if (std::abs(A) < tiny) {
    if (std::abs(B) < tiny) {
      if (C < 0.0) push(t0, t1, -1, -1);
    } else {
      const double r = -C / B;
      if (B > 0.0) push(t0, r, -1, tag);
      else push(r, t1, tag, -1);
    }
    return;
  }
  const double disc = B * B - 4.0 * A * C;
  if (A > 0.0) {
    if (disc <= 0.0) return;
    const double s = std::sqrt(disc);
    // Numerically stable roots.
    const double qq = -0.5 * (B + std::copysign(s, B));
    double r1 = qq / A, r2 = (qq != 0.0) ? C / qq : r1;
    if (r1 > r2) std::swap(r1, r2);
    push(r1, r2, tag, tag);
  } else {
    if (disc <= 0.0) { push(t0, t1, -1, -1); return; }
    const double s = std::sqrt(disc);
    const double qq = -0.5 * (B + std::copysign(s, B));
    double r1 = qq / A, r2 = (qq != 0.0) ? C / qq : r1;
    if (r1 > r2) std::swap(r1, r2);
    push(t0, r1, -1, tag);
    push(r2, t1, tag, -1);
  }
}

// ---------------------------------------------------------------------------
// CSG region.
// ---------------------------------------------------------------------------
struct Region {
  enum class Op { kPrim, kNot, kAnd, kOr };
  struct Node {
    Op op;
    int a = -1, b = -1;   // children (node indices) or primitive index for kPrim
  };
  std::vector<Quadric> prims;
  std::vector<Node> nodes;
  int root = -1;

  int prim(const Quadric& q) {
    prims.push_back(q);
    nodes.push_back({Op::kPrim, static_cast<int>(prims.size()) - 1, -1});
    return static_cast<int>(nodes.size()) - 1;
  }
  int opNot(int x) { nodes.push_back({Op::kNot, x, -1}); return static_cast<int>(nodes.size()) - 1; }
  int opAnd(int x, int y) { nodes.push_back({Op::kAnd, x, y}); return static_cast<int>(nodes.size()) - 1; }
  int opOr(int x, int y) { nodes.push_back({Op::kOr, x, y}); return static_cast<int>(nodes.size()) - 1; }

  bool inside(const Vec3& x) const { return insideNode(root, x); }
  bool insideNode(int n, const Vec3& x) const {
    const Node& nd = nodes[n];
    switch (nd.op) {
      case Op::kPrim: return prims[nd.a].eval(x) < 0.0;
      case Op::kNot: return !insideNode(nd.a, x);
      case Op::kAnd: return insideNode(nd.a, x) && insideNode(nd.b, x);
      case Op::kOr: return insideNode(nd.a, x) || insideNode(nd.b, x);
    }
    return false;
  }

  void column(const Vec3& base, int axis, double t0, double t1, IntervalList* out) const {
    columnNode(root, base, axis, t0, t1, out);
  }
  void columnNode(int n, const Vec3& base, int axis, double t0, double t1, IntervalList* out) const {
    const Node& nd = nodes[n];
    switch (nd.op) {
      case Op::kPrim:
        quadricIntervals(prims[nd.a], nd.a, base, axis, t0, t1, out);
        return;
      case Op::kNot: {
        IntervalList tmp;
        columnNode(nd.a, base, axis, t0, t1, &tmp);
        complementList(tmp, t0, t1, out);
        return;
      }
      case Op::kAnd:
      case Op::kOr: {
        IntervalList l, r;
        columnNode(nd.a, base, axis, t0, t1, &l);
        columnNode(nd.b, base, axis, t0, t1, &r);
        if (nd.op == Op::kAnd) intersectLists(l, r, out);
        else unionLists(l, r, out);
        return;
      }
    }
  }

  // Applies the same rigid transform to every primitive.
  void transform(const Mat3& R, const Vec3& shift, const Vec3& anchor) {
    for (auto& q : prims) q = q.transformed(R, shift, anchor);
  }
};

// ---------------------------------------------------------------------------
// Cell integration.
// ---------------------------------------------------------------------------
struct CellMoments {
  double vf = 0.0;
  Vec3 liq{};   // global liquid centroid (cell centre if no liquid)
  Vec3 gas{};   // global gas centroid (cell centre if no gas)
};

// Per-primitive surface patch statistics inside one cell.
struct SurfacePatch {
  int prim = -1;
  double area = 0.0;
  Vec3 normal_sum{};    // integral of outward unit normal dA
  Vec3 centroid_sum{};  // integral of x dA
  Vec3 normal() const { return normalized(normal_sum); }
  Vec3 centroid() const { return area > 0.0 ? (1.0 / area) * centroid_sum : Vec3{0, 0, 0}; }
};

// True if every primitive surface provably misses the box (conservative).
inline bool cellIsUniform(const Region& reg, const Vec3& lo, const Vec3& hi) {
  const Vec3 c = 0.5 * (lo + hi);
  const Vec3 h = hi - lo;
  const double r = 0.5 * norm(h);
  for (const auto& q : reg.prims) {
    const double v = q.eval(c);
    const Vec3 g = q.grad(c);
    const double bound = norm(g) * r + q.frobA() * r * r;
    if (std::abs(v) <= bound) return false;
  }
  return true;
}

// Picks the column axis best aligned with the normal of the nearest surface.
inline int chooseAxis(const Region& reg, const Vec3& c) {
  double best = 1.0e300;
  Vec3 n{0.0, 0.0, 1.0};
  for (const auto& q : reg.prims) {
    const Vec3 g = q.grad(c);
    const double gn = norm(g);
    if (gn <= 0.0) continue;
    const double dist = std::abs(q.eval(c)) / gn;
    if (dist < best) { best = dist; n = g; }
  }
  int axis = 0;
  for (int a = 1; a < 3; ++a)
    if (std::abs(n[a]) > std::abs(n[axis])) axis = a;
  return axis;
}

// Volume moments of the region in the box [lo,hi] using nq x nq columns.
inline CellMoments integrateCell(const Region& reg, const Vec3& lo, const Vec3& hi, int nq) {
  CellMoments m;
  const Vec3 cc = 0.5 * (lo + hi);
  const double vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  if (cellIsUniform(reg, lo, hi)) {
    m.vf = reg.inside(cc) ? 1.0 : 0.0;
    m.liq = cc;
    m.gas = cc;
    return m;
  }
  const int axis = chooseAxis(reg, cc);
  const int u = (axis + 1) % 3, v = (axis + 2) % 3;
  const double hu = (hi[u] - lo[u]) / nq, hv = (hi[v] - lo[v]) / nq;
  double V = 0.0;
  Vec3 M{};
  IntervalList ivs;
  for (int a = 0; a < nq; ++a) {
    for (int b = 0; b < nq; ++b) {
      Vec3 base{};
      base[u] = lo[u] + (a + 0.5) * hu;
      base[v] = lo[v] + (b + 0.5) * hv;
      base[axis] = 0.0;
      reg.column(base, axis, lo[axis], hi[axis], &ivs);
      for (const auto& iv : ivs) {
        const double len = iv.hi - iv.lo;
        V += len;
        M[u] += len * base[u];
        M[v] += len * base[v];
        M[axis] += 0.5 * (iv.hi * iv.hi - iv.lo * iv.lo);
      }
    }
  }
  V *= hu * hv;
  M = (hu * hv) * M;
  m.vf = std::min(1.0, std::max(0.0, V / vol));
  m.liq = V > 0.0 ? (1.0 / V) * M : cc;
  const double Vg = vol - V;
  m.gas = Vg > 1.0e-300 ? (1.0 / Vg) * (vol * cc - M) : cc;
  return m;
}

// Surface patches of the region boundary inside the box, one per primitive.
// Integrates along all three axes with weight |n_axis| h^2, which sums to the
// exact surface integral (sum_axis n_axis^2 = 1) and never divides by a
// grazing normal component.
inline std::vector<SurfacePatch> surfacePatches(const Region& reg, const Vec3& lo, const Vec3& hi, int nq) {
  std::vector<SurfacePatch> patches(reg.prims.size());
  for (std::size_t p = 0; p < patches.size(); ++p) patches[p].prim = static_cast<int>(p);
  IntervalList ivs;
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    const double hu = (hi[u] - lo[u]) / nq, hv = (hi[v] - lo[v]) / nq;
    for (int a = 0; a < nq; ++a) {
      for (int b = 0; b < nq; ++b) {
        Vec3 base{};
        base[u] = lo[u] + (a + 0.5) * hu;
        base[v] = lo[v] + (b + 0.5) * hv;
        reg.column(base, axis, lo[axis], hi[axis], &ivs);
        for (const auto& iv : ivs) {
          for (int end = 0; end < 2; ++end) {
            const int tag = end == 0 ? iv.tag_lo : iv.tag_hi;
            if (tag < 0) continue;
            Vec3 x = base;
            x[axis] = end == 0 ? iv.lo : iv.hi;
            Vec3 n = normalized(reg.prims[tag].grad(x));
            // Outward: the region lies above a lower endpoint and below an
            // upper one along the column.
            const double want = end == 0 ? -1.0 : 1.0;
            if (n[axis] * want < 0.0) n = -1.0 * n;
            const double w = std::abs(n[axis]) * hu * hv;
            patches[tag].area += w;
            patches[tag].normal_sum = patches[tag].normal_sum + w * n;
            patches[tag].centroid_sum = patches[tag].centroid_sum + w * x;
          }
        }
      }
    }
  }
  return patches;
}

}  // namespace bench

#endif  // EXAMPLES_R2PNET_BENCH_GEOMETRY_H_
