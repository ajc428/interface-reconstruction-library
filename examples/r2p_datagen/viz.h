// Optional ParaView output (legacy ASCII VTK), a few samples per family.
//
// Per sample, in grid coordinates (centre cell = [-0.5, 0.5]^3):
//   <tag>_surface.vtk    the exact interface over the stencil, triangulated;
//                        cell data: face (0/1), centre (1 inside the centre cell)
//   <tag>_cells.vtk      the stencil cells; cell data: vf (liquid), centre
//   <tag>_centroids.vtk  liquid and gas centroids of mixed cells as the network
//                        sees them (after noise); point data: phase (0 liquid,
//                        1 gas), vf
//   <tag>_faces.vtk      one point per labelled face at its centroid in the
//                        centre cell; point data: normal (out of the liquid),
//                        area, face

#ifndef EXAMPLES_R2P_DATAGEN_VIZ_H_
#define EXAMPLES_R2P_DATAGEN_VIZ_H_

#include <cstdio>
#include <string>
#include <vector>

#include "examples/r2p_datagen/stencil.h"

namespace r2pgen {

struct VizTri {
  Vec3 p[3];
  Vec3 outward;   // out of the liquid
  int face = 0;
  int centre = 0;
};

// Triangulates the scene's interface cell by cell over the N^3 stencil.
inline std::vector<VizTri> triangulateScene(const Scene& s, int N, double length) {
  std::vector<VizTri> out;
  for (int c = 0; c < N * N * N; ++c) {
    const Vec3 cc = cellCentre(N, c);
    const IRL::RectangularCuboid cell = unitCell(cc);
    const int centre = cc.cwiseAbs().maxCoeff() < 0.25 ? 1 : 0;
    auto addSurface = [&](const Para& p, int face, double sign) {
      const auto tri =
          IRL::getVolumeMoments<IRL::AddSurfaceOutput<IRL::VolumeMoments, IRL::ParaboloidParametrizedSurfaceOutput>>(
              cell, toIRL(p))
              .getSurface()
              .triangulate(length);
      for (const auto& t : tri.getTriangleList()) {
        VizTri v;
        for (int k = 0; k < 3; ++k) v.p[k] = Vec3(t[k][0], t[k][1], t[k][2]);
        const Vec3 mid = (v.p[0] + v.p[1] + v.p[2]) / 3.0;
        v.outward = sign * p.outward(mid);
        v.face = face;
        if (face < 0) v.face = (mid - p.d).dot(s.split_axis == 0 ? p.e0 : p.e1) >= 0.0 ? 1 : 0;
        v.centre = centre;
        out.push_back(v);
      }
    };
    switch (s.kind) {
      case Kind::kSingle: addSurface(s.upper, s.split_axis >= 0 ? -1 : 0, 1.0); break;
      case Kind::kNested: addSurface(s.lower, 0, -1.0); addSurface(s.upper, 1, 1.0); break;
      case Kind::kPlanes: {
        const IRL::PlanarSeparator sep = toIRL(s.plane);
        for (int k = 0; k < 2; ++k) {
          const auto poly = IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, sep, sep[k]);
          const int nv = int(poly.getNumberOfVertices());
          for (int v = 1; v + 1 < nv; ++v) {
            VizTri t;
            t.p[0] = Vec3(poly[0][0], poly[0][1], poly[0][2]);
            t.p[1] = Vec3(poly[v][0], poly[v][1], poly[v][2]);
            t.p[2] = Vec3(poly[v + 1][0], poly[v + 1][1], poly[v + 1][2]);
            t.outward = s.plane[k].n;
            t.face = k;
            t.centre = centre;
            out.push_back(t);
          }
        }
        break;
      }
    }
  }
  return out;
}

inline void writeViz(const std::string& tag, const Scene& s, const std::vector<double>& moments, int N,
                     double length) {
  const std::vector<VizTri> tris = triangulateScene(s, N, length);
  if (std::FILE* probe = std::fopen((tag + "_surface.vtk").c_str(), "w")) {
    std::fclose(probe);
  } else {
    std::fprintf(stderr, "viz: cannot write %s_*.vtk\n", tag.c_str());
    return;
  }

  if (std::FILE* f = std::fopen((tag + "_surface.vtk").c_str(), "w")) {
    std::fprintf(f, "# vtk DataFile Version 3.0\nr2p_datagen surface\nASCII\nDATASET POLYDATA\nPOINTS %zu double\n",
                 3 * tris.size());
    for (const auto& t : tris)
      for (const auto& p : t.p) std::fprintf(f, "%.9g %.9g %.9g\n", p.x(), p.y(), p.z());
    std::fprintf(f, "POLYGONS %zu %zu\n", tris.size(), 4 * tris.size());
    for (std::size_t i = 0; i < tris.size(); ++i) std::fprintf(f, "3 %zu %zu %zu\n", 3 * i, 3 * i + 1, 3 * i + 2);
    std::fprintf(f, "CELL_DATA %zu\nSCALARS face int 1\nLOOKUP_TABLE default\n", tris.size());
    for (const auto& t : tris) std::fprintf(f, "%d\n", t.face);
    std::fprintf(f, "SCALARS centre int 1\nLOOKUP_TABLE default\n");
    for (const auto& t : tris) std::fprintf(f, "%d\n", t.centre);
    std::fclose(f);
  }

  const int ncell = N * N * N;
  if (std::FILE* f = std::fopen((tag + "_cells.vtk").c_str(), "w")) {
    std::fprintf(f, "# vtk DataFile Version 3.0\nr2p_datagen cells\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS %d double\n",
                 8 * ncell);
    for (int c = 0; c < ncell; ++c) {
      const Vec3 cc = cellCentre(N, c);
      for (int k = 0; k < 8; ++k) {   // VTK hexahedron vertex order
        const double dx = (k == 1 || k == 2 || k == 5 || k == 6) ? 0.5 : -0.5;
        const double dy = (k == 2 || k == 3 || k == 6 || k == 7) ? 0.5 : -0.5;
        const double dz = k >= 4 ? 0.5 : -0.5;
        std::fprintf(f, "%g %g %g\n", cc.x() + dx, cc.y() + dy, cc.z() + dz);
      }
    }
    std::fprintf(f, "CELLS %d %d\n", ncell, 9 * ncell);
    for (int c = 0; c < ncell; ++c) {
      std::fprintf(f, "8");
      for (int k = 0; k < 8; ++k) std::fprintf(f, " %d", 8 * c + k);
      std::fprintf(f, "\n");
    }
    std::fprintf(f, "CELL_TYPES %d\n", ncell);
    for (int c = 0; c < ncell; ++c) std::fprintf(f, "12\n");
    std::fprintf(f, "CELL_DATA %d\nSCALARS vf double 1\nLOOKUP_TABLE default\n", ncell);
    for (int c = 0; c < ncell; ++c) std::fprintf(f, "%.9g\n", moments[7 * c]);
    std::fprintf(f, "SCALARS centre int 1\nLOOKUP_TABLE default\n");
    for (int c = 0; c < ncell; ++c) std::fprintf(f, "%d\n", c == ncell / 2 ? 1 : 0);
    std::fclose(f);
  }

  if (std::FILE* f = std::fopen((tag + "_centroids.vtk").c_str(), "w")) {
    std::vector<std::pair<Vec3, int>> pts;
    std::vector<double> vf;
    for (int c = 0; c < ncell; ++c) {
      const double* o = &moments[7 * c];
      if (o[0] <= 0.0 || o[0] >= 1.0) continue;
      const Vec3 cc = cellCentre(N, c);
      pts.push_back({cc + Vec3(o[1], o[2], o[3]), 0});
      pts.push_back({cc + Vec3(o[4], o[5], o[6]), 1});
      vf.push_back(o[0]);
      vf.push_back(o[0]);
    }
    std::fprintf(f, "# vtk DataFile Version 3.0\nr2p_datagen centroids\nASCII\nDATASET POLYDATA\nPOINTS %zu double\n",
                 pts.size());
    for (const auto& p : pts) std::fprintf(f, "%.9g %.9g %.9g\n", p.first.x(), p.first.y(), p.first.z());
    std::fprintf(f, "VERTICES %zu %zu\n", pts.size(), 2 * pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) std::fprintf(f, "1 %zu\n", i);
    std::fprintf(f, "POINT_DATA %zu\nSCALARS phase int 1\nLOOKUP_TABLE default\n", pts.size());
    for (const auto& p : pts) std::fprintf(f, "%d\n", p.second);
    std::fprintf(f, "SCALARS vf double 1\nLOOKUP_TABLE default\n");
    for (double v : vf) std::fprintf(f, "%.9g\n", v);
    std::fclose(f);
  }

  // Face centroids and area-averaged normals in the centre cell.
  Vec3 sum_n[2] = {Vec3::Zero(), Vec3::Zero()}, sum_x[2] = {Vec3::Zero(), Vec3::Zero()};
  double area[2] = {0.0, 0.0};
  for (const auto& t : tris) {
    if (!t.centre) continue;
    const double a = 0.5 * (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]).norm();
    sum_n[t.face] += a * t.outward;
    sum_x[t.face] += a * (t.p[0] + t.p[1] + t.p[2]) / 3.0;
    area[t.face] += a;
  }
  if (std::FILE* f = std::fopen((tag + "_faces.vtk").c_str(), "w")) {
    std::vector<int> faces;
    for (int k = 0; k < 2; ++k) if (area[k] > 0.0) faces.push_back(k);
    std::fprintf(f, "# vtk DataFile Version 3.0\nr2p_datagen faces\nASCII\nDATASET POLYDATA\nPOINTS %zu double\n",
                 faces.size());
    for (int k : faces) {
      const Vec3 x = sum_x[k] / area[k];
      std::fprintf(f, "%.9g %.9g %.9g\n", x.x(), x.y(), x.z());
    }
    std::fprintf(f, "VERTICES %zu %zu\n", faces.size(), 2 * faces.size());
    for (std::size_t i = 0; i < faces.size(); ++i) std::fprintf(f, "1 %zu\n", i);
    std::fprintf(f, "POINT_DATA %zu\nVECTORS normal double\n", faces.size());
    for (int k : faces) {
      const Vec3 n = sum_n[k].normalized();
      std::fprintf(f, "%.9g %.9g %.9g\n", n.x(), n.y(), n.z());
    }
    std::fprintf(f, "SCALARS area double 1\nLOOKUP_TABLE default\n");
    for (int k : faces) std::fprintf(f, "%.9g\n", area[k]);
    std::fprintf(f, "SCALARS face int 1\nLOOKUP_TABLE default\n");
    for (int k : faces) std::fprintf(f, "%d\n", k);
    std::fclose(f);
  }
}

}  // namespace r2pgen

#endif  // EXAMPLES_R2P_DATAGEN_VIZ_H_
