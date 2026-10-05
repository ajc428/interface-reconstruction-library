// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2026 Andrew Cahaly <andrew.cahaly@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Advects a liquid sphere through a prescribed velocity field, reconstructing
// the interface with R2P-Net (r2p_net.h) every step.
//
//   r2p_advector <Film3D|Deformation3D> <cells> <dt> <periods> [output_every]
//
// Film3D flattens the sphere into a sheet across the (1,1,1) diagonal;
// Deformation3D stretches it into sheets and returns it after one period.
// Volume fractions and centroids are transported fully Lagrangian: each cell
// is traced back through the velocity field, cut by the previous interface,
// and its phase centroids are moved forward. Output (every output_every steps,
// and the first and last): viz/vf_*.vtk and viz/interface_*.vtu.

#include <sys/stat.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/polyhedrons/dodecahedron.h"
#include "irl/moments/volume_moments_and_normal.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/localized_separator_link.h"
#include "irl/planar_reconstruction/planar_localizer.h"

#include "examples/r2p_advector/grid.h"
#include "examples/r2p_advector/r2p_net.h"

namespace {

// ===========================================================================
// Test cases: a liquid sphere in a periodic unit box
// ===========================================================================

struct TestCase {
  const char* name;
  double period;          // simulated time per period
  IRL::Pt centre;         // initial sphere
  double radius;
  IRL::Vec3<double> (*velocity)(const IRL::Pt& x, double t, const Grid& grid);
};

// Uniaxial compression along n = (1,1,1)/sqrt(3) about the drop's centre, with
// stretching in the plane normal to it: the drop thins into a sheet.
IRL::Vec3<double> filmVelocity(const IRL::Pt& x, const double, const Grid& grid) {
  const double rate = 1.0;
  const double magnitude = std::sqrt(3.0), n[3] = {1.0 / magnitude, 1.0 / magnitude, 1.0 / magnitude};
  const double r[3] = {x[0] - 0.5, x[1] - 0.5, x[2] - (0.5 + grid.h[2] / 2.0)};
  const double along = r[0] * n[0] + r[1] * n[1] + r[2] * n[2];
  return IRL::Vec3<double>(rate * (r[0] - 3.0 * along * n[0]), rate * (r[1] - 3.0 * along * n[1]),
                           rate * (r[2] - 3.0 * along * n[2]));
}

// LeVeque's 3D deformation field, reversed at half the period of 3.
IRL::Vec3<double> deformationVelocity(const IRL::Pt& x, const double t, const Grid&) {
  const double s = std::cos(M_PI * t / 3.0);
  return IRL::Vec3<double>(
      2.0 * std::pow(std::sin(M_PI * x[0]), 2) * std::sin(2.0 * M_PI * x[1]) * std::sin(2.0 * M_PI * x[2]) * s,
      -std::sin(2.0 * M_PI * x[0]) * std::pow(std::sin(M_PI * x[1]), 2) * std::sin(2.0 * M_PI * x[2]) * s,
      -std::sin(2.0 * M_PI * x[0]) * std::sin(2.0 * M_PI * x[1]) * std::pow(std::sin(M_PI * x[2]), 2) * s);
}

bool findCase(const std::string& name, const Grid& grid, TestCase* found) {
  if (name == "Film3D") {
    *found = {"Film3D", 1.0, IRL::Pt(0.5, 0.5, 0.5 + grid.h[2] / 2.0), 0.125, filmVelocity};
    return true;
  }
  if (name == "Deformation3D") {
    *found = {"Deformation3D", 3.0, IRL::Pt(0.35, 0.35, 0.35), 0.15, deformationVelocity};
    return true;
  }
  return false;
}

// Initial interface: one plane per cell near the sphere, tangent to it at the
// point facing the cell centre; pure cells further than 3 cells away.
void initialiseSphere(const TestCase& test, Field<IRL::PlanarSeparator>* interface) {
  const Grid& grid = interface->grid();
  FOR_INTERIOR(grid, i, j, k) {
    const IRL::RectangularCuboid cell = grid.cell(i, j, k);
    const double distance = IRL::magnitude(cell.calculateCentroid() - test.centre) - test.radius;
    IRL::PlanarSeparator& sep = (*interface)(i, j, k);
    if (std::abs(distance) > 3.0 * grid.h[0]) {
      sep = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), distance > 0.0 ? -1.0e7 : 1.0e7));
      continue;
    }
    const IRL::Normal n = IRL::Normal::fromPtNormalized(cell.calculateCentroid() - test.centre);
    const IRL::PlanarSeparator tangent = IRL::PlanarSeparator::fromOnePlane(
        IRL::Plane(n, n * IRL::Pt(test.centre + IRL::Normal::toPt(n * test.radius))));
    auto moments = IRL::getVolumeMoments<IRL::VolumeMomentsAndNormal>(
        IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, tangent, tangent[0]));
    if (moments.volumeMoments().volume() == 0.0) {   // the plane misses the cell
      sep = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(IRL::Normal(0.0, 0.0, 0.0), std::copysign(1.0, -distance)));
      continue;
    }
    moments.normalizeByVolume();
    moments.normal().normalize();
    sep = IRL::PlanarSeparator::fromOnePlane(
        IRL::Plane(moments.normal(), moments.normal() * moments.volumeMoments().centroid()));
  }
  r2p::fillGhostPlanes(interface);
}

// ===========================================================================
// Transport
// ===========================================================================

struct Velocity {
  explicit Velocity(const Grid& grid) : u(grid), v(grid), w(grid) {}
  Field<double> u, v, w;   // sampled at cell centres, ghosts included

  void sample(const TestCase& test, const double t) {
    const Grid& grid = u.grid();
    FOR_ALL(grid, i, j, k) {
      const IRL::Vec3<double> velocity = test.velocity(grid.cellCentre(i, j, k), t, grid);
      u(i, j, k) = velocity[0];
      v(i, j, k) = velocity[1];
      w(i, j, k) = velocity[2];
    }
  }
  IRL::Vec3<double> at(const IRL::Pt& x) const {
    return IRL::Vec3<double>(u.interpolate(x), v.interpolate(x), w.interpolate(x));
  }
  // Point moved for a time dt (negative: traced back), RK4.
  IRL::Pt move(const IRL::Pt& x, const double dt) const {
    const auto v1 = at(x);
    const auto v2 = at(x + IRL::Pt::fromVec3(0.5 * dt * v1));
    const auto v3 = at(x + IRL::Pt::fromVec3(0.5 * dt * v2));
    const auto v4 = at(x + IRL::Pt::fromVec3(dt * v3));
    return x + IRL::Pt::fromVec3(dt * (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0);
  }
};

// Phase fields: liquid volume fraction and the liquid and gas centroids.
struct PhaseFields {
  explicit PhaseFields(const Grid& grid) : vf(grid), liquid_centroid(grid), gas_centroid(grid) {}
  Field<double> vf;
  Field<IRL::Pt> liquid_centroid, gas_centroid;

  // Ghost cells: periodic copies, centroids shifted by one domain length.
  void fillGhosts() {
    vf.fillGhosts();
    liquid_centroid.fillGhosts();
    gas_centroid.fillGhosts();
    const Grid& grid = vf.grid();
    FOR_ALL(grid, i, j, k) {
      if (!grid.isGhost(i, j, k)) continue;
      const int index[3] = {i, j, k};
      for (int d = 0; d < 3; ++d) {
        const double shift = index[d] < 0 ? -grid.length(d) : (index[d] >= grid.n[d] ? grid.length(d) : 0.0);
        liquid_centroid(i, j, k)[d] += shift;
        gas_centroid(i, j, k)[d] += shift;
      }
    }
  }
};

// Volume fractions and centroids of the reconstructed interface, every cell.
void phaseFieldsFromInterface(const Field<IRL::PlanarSeparator>& interface, PhaseFields* phases) {
  const Grid& grid = interface.grid();
  FOR_ALL(grid, i, j, k) {
    const IRL::RectangularCuboid cell = grid.cell(i, j, k);
    const auto moments =
        IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(cell, interface(i, j, k));
    phases->vf(i, j, k) = moments[0].volume() / cell.calculateVolume();
    phases->liquid_centroid(i, j, k) = moments[0].centroid();
    phases->gas_centroid(i, j, k) = moments[1].centroid();
  }
}

// Each cell's planes, localised to the cell and linked to its neighbours, so a
// traced-back cell can be cut by the interface of every cell it overlaps.
struct LinkedInterface {
  LinkedInterface(const Grid& grid, Field<IRL::PlanarSeparator>* interface) : localizers(grid), links(grid) {
    FOR_ALL(grid, i, j, k) {
      localizers(i, j, k) = grid.cell(i, j, k).getLocalizer();
      links(i, j, k) = IRL::LocalizedSeparatorLink(&localizers(i, j, k), &(*interface)(i, j, k));
    }
    auto link_or_null = [&](const int i, const int j, const int k) {
      const bool outside = i < -grid.ghosts || j < -grid.ghosts || k < -grid.ghosts ||
                           i >= grid.n[0] + grid.ghosts || j >= grid.n[1] + grid.ghosts || k >= grid.n[2] + grid.ghosts;
      return outside ? nullptr : &links(i, j, k);
    };
    FOR_ALL(grid, i, j, k) {
      IRL::LocalizedSeparatorLink& link = links(i, j, k);
      link.setId(static_cast<IRL::UnsignedIndex_t>(grid.index(i, j, k)));
      link.setEdgeConnectivity(0, link_or_null(i - 1, j, k));
      link.setEdgeConnectivity(1, link_or_null(i + 1, j, k));
      link.setEdgeConnectivity(2, link_or_null(i, j - 1, k));
      link.setEdgeConnectivity(3, link_or_null(i, j + 1, k));
      link.setEdgeConnectivity(4, link_or_null(i, j, k - 1));
      link.setEdgeConnectivity(5, link_or_null(i, j, k + 1));
    }
  }
  Field<IRL::PlanarLocalizer> localizers;
  Field<IRL::LocalizedSeparatorLink> links;
};

// One step: every cell traced back over dt and cut by the current interface.
void advect(const Velocity& velocity, const double dt, const LinkedInterface& linked, PhaseFields* phases) {
  const Grid& grid = phases->vf.grid();
  FOR_INTERIOR(grid, i, j, k) {
    const IRL::RectangularCuboid cell = grid.cell(i, j, k);
    IRL::Dodecahedron traced_cell;
    for (IRL::UnsignedIndex_t v = 0; v < 8; ++v) traced_cell[v] = velocity.move(cell[v], -dt);
    const auto moments = IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(
        traced_cell, linked.links(i, j, k));
    double& vf = phases->vf(i, j, k);
    IRL::Pt &liquid = phases->liquid_centroid(i, j, k), &gas = phases->gas_centroid(i, j, k);
    vf = moments[0].volume() / (moments[0].volume() + moments[1].volume());
    if (vf < IRL::global_constants::VF_LOW || vf > IRL::global_constants::VF_HIGH) {
      vf = vf < IRL::global_constants::VF_LOW ? 0.0 : 1.0;
      liquid = gas = cell.calculateCentroid();
    } else {   // the traced-back centroids, moved forward again
      liquid = velocity.move(moments[0].centroid(), dt);
      gas = velocity.move(moments[1].centroid(), dt);
    }
  }
  phases->fillGhosts();
}

// ===========================================================================
// Output
// ===========================================================================

std::string numbered(const char* stem, const int number, const char* extension) {
  char name[64];
  std::snprintf(name, sizeof(name), "viz/%s_%06d.%s", stem, number, extension);
  return name;
}

void writeVolumeFraction(const Field<double>& vf, const int number) {
  const Grid& grid = vf.grid();
  FILE* file = std::fopen(numbered("vf", number, "vtk").c_str(), "w");
  std::fprintf(file, "# vtk DataFile Version 3.0\nvolume fraction\nASCII\nDATASET RECTILINEAR_GRID\n");
  std::fprintf(file, "DIMENSIONS %d %d %d\n", grid.n[0] + 1, grid.n[1] + 1, grid.n[2] + 1);
  const char* axes[3] = {"X", "Y", "Z"};
  for (int d = 0; d < 3; ++d) {
    std::fprintf(file, "%s_COORDINATES %d float\n", axes[d], grid.n[d] + 1);
    for (int c = 0; c <= grid.n[d]; ++c) std::fprintf(file, "%.8e\n", grid.face(d, c));
  }
  std::fprintf(file, "CELL_DATA %d\nSCALARS VolumeFraction double 1\nLOOKUP_TABLE default\n",
               grid.n[0] * grid.n[1] * grid.n[2]);
  for (int k = 0; k < grid.n[2]; ++k)
    for (int j = 0; j < grid.n[1]; ++j)
      for (int i = 0; i < grid.n[0]; ++i) std::fprintf(file, "%.10e\n", vf(i, j, k));
  std::fclose(file);
}

// Interface polygons, with each cell's diagnostics as polygon data.
void writeInterface(const Field<IRL::PlanarSeparator>& interface, const Field<double>& vf,
                    const r2p::Diagnostics& diagnostics, const int number) {
  const Grid& grid = interface.grid();
  std::string points, connectivity, offsets;
  struct Tagged {
    int i, j, k, planes;
  };
  std::vector<Tagged> polygons;
  long n_points = 0;
  char buffer[128];
  FOR_INTERIOR(grid, i, j, k) {
    const double f = vf(i, j, k);
    if (f < IRL::global_constants::VF_LOW || f > IRL::global_constants::VF_HIGH) continue;
    const IRL::PlanarSeparator& sep = interface(i, j, k);
    const IRL::RectangularCuboid cell = grid.cell(i, j, k);
    for (IRL::UnsignedIndex_t p = 0; p < sep.getNumberOfPlanes(); ++p) {
      const IRL::Polygon polygon = IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, sep, sep[p]);
      if (polygon.getNumberOfVertices() < 3) continue;
      for (IRL::UnsignedIndex_t v = 0; v < polygon.getNumberOfVertices(); ++v) {
        std::snprintf(buffer, sizeof(buffer), "%.8e %.8e %.8e\n", polygon[v][0], polygon[v][1], polygon[v][2]);
        points += buffer;
        connectivity += std::to_string(n_points++) + " ";
      }
      offsets += std::to_string(n_points) + " ";
      polygons.push_back({i, j, k, static_cast<int>(sep.getNumberOfPlanes())});
    }
  }
  FILE* file = std::fopen(numbered("interface", number, "vtu").c_str(), "w");
  std::fprintf(file, "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                     "<UnstructuredGrid>\n<Piece NumberOfPoints=\"%ld\" NumberOfCells=\"%zu\">\n",
               n_points, polygons.size());
  std::fprintf(file, "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n%s</DataArray>\n</Points>\n",
               points.c_str());
  std::fprintf(file, "<Cells>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n%s\n</DataArray>\n"
                     "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n%s\n</DataArray>\n"
                     "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n",
               connectivity.c_str(), offsets.c_str());
  for (std::size_t q = 0; q < polygons.size(); ++q) std::fprintf(file, "7 ");
  std::fprintf(file, "\n</DataArray>\n</Cells>\n<CellData>\n");
  auto write_int = [&](const char* name, auto value) {
    std::fprintf(file, "<DataArray type=\"Int32\" Name=\"%s\" format=\"ascii\">\n", name);
    for (const Tagged& t : polygons) std::fprintf(file, "%d ", value(t));
    std::fprintf(file, "\n</DataArray>\n");
  };
  write_int("planes", [](const Tagged& t) { return t.planes; });
  write_int("route", [&](const Tagged& t) { return diagnostics.route(t.i, t.j, t.k); });
  write_int("class", [&](const Tagged& t) { return diagnostics.classifier(t.i, t.j, t.k); });
  write_int("guard", [&](const Tagged& t) { return diagnostics.guard(t.i, t.j, t.k); });
  write_int("slab", [&](const Tagged& t) { return diagnostics.slab(t.i, t.j, t.k); });
  write_int("edge", [&](const Tagged& t) { return diagnostics.edge(t.i, t.j, t.k); });
  std::fprintf(file, "<DataArray type=\"Float64\" Name=\"unpinch\" format=\"ascii\">\n");
  for (const Tagged& t : polygons) std::fprintf(file, "%g ", diagnostics.unpinch(t.i, t.j, t.k));
  std::fprintf(file, "\n</DataArray>\n</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n");
  std::fclose(file);
}

// Liquid volume (in cell volumes), mixed cells and two-plane cells.
struct Census {
  double liquid = 0.0;
  int mixed = 0, two_plane = 0;
};
Census census(const Field<double>& vf, const Field<IRL::PlanarSeparator>& interface) {
  Census c;
  FOR_INTERIOR(vf.grid(), i, j, k) {
    c.liquid += vf(i, j, k);
    if (vf(i, j, k) < IRL::global_constants::VF_LOW || vf(i, j, k) > IRL::global_constants::VF_HIGH) continue;
    ++c.mixed;
    if (interface(i, j, k).getNumberOfPlanes() == 2) ++c.two_plane;
  }
  return c;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 5 || argc > 6) {
    std::printf("Usage: %s <Film3D|Deformation3D> <cells> <dt> <periods> [output_every]\n", argv[0]);
    return 1;
  }
  const int cells = std::atoi(argv[2]);
  const double dt = std::atof(argv[3]);
  const double periods = std::atof(argv[4]);
  const int output_every = argc == 6 ? std::atoi(argv[5]) : 0;

  const Grid grid(cells, 2, IRL::Pt(0.0, 0.0, 0.0), IRL::Pt(1.0, 1.0, 1.0));
  TestCase test;
  if (!findCase(argv[1], grid, &test)) {
    std::printf("Unknown case %s (Film3D or Deformation3D)\n", argv[1]);
    return 1;
  }
  const double end_time = periods * test.period;
  IRL::setMinimumVolumeToTrack(10.0 * DBL_EPSILON * grid.h[0] * grid.h[1] * grid.h[2]);
  IRL::setVolumeFractionBounds(1.0e-8);
  IRL::setVolumeFractionTolerance(1.0e-13);

  Field<IRL::PlanarSeparator> interface(grid);
  PhaseFields phases(grid);
  Velocity velocity(grid);
  r2p::Diagnostics diagnostics(grid);
  const LinkedInterface linked(grid, &interface);
  initialiseSphere(test, &interface);
  phaseFieldsFromInterface(interface, &phases);
  const Field<double> initial_vf = phases.vf;

  mkdir("viz", 0777);
  int output_number = 0;
  if (output_every > 0) {
    writeVolumeFraction(phases.vf, output_number);
    writeInterface(interface, phases.vf, diagnostics, output_number++);
  }
  const Census start = census(phases.vf, interface);
  std::printf("%6s %10s %16s %8s %10s %12s %12s\n", "step", "time", "volume change", "mixed", "two-plane",
              "advect [s]", "recon [s]");

  using Clock = std::chrono::steady_clock;
  const auto run_start = Clock::now();
  double time = 0.0, advect_seconds = 0.0, reconstruct_seconds = 0.0;
  int step = 0;
  while (time < end_time) {
    const double step_dt = std::fmin(dt, end_time - time);
    velocity.sample(test, time + 0.5 * step_dt);
    const auto t0 = Clock::now();
    advect(velocity, step_dt, linked, &phases);
    const auto t1 = Clock::now();
    r2p::reconstruct(phases.vf, phases.liquid_centroid, phases.gas_centroid, &interface, &diagnostics);
    const auto t2 = Clock::now();
    advect_seconds += std::chrono::duration<double>(t1 - t0).count();
    reconstruct_seconds += std::chrono::duration<double>(t2 - t1).count();
    time += step_dt;
    ++step;

    const Census now = census(phases.vf, interface);
    std::printf("%6d %10.4f %16.6e %8d %10d %12.4f %12.4f\n", step, time, (now.liquid - start.liquid) / start.liquid,
                now.mixed, now.two_plane, std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count());
    if (output_every > 0 && (step % output_every == 0 || time >= end_time)) {
      writeVolumeFraction(phases.vf, output_number);
      writeInterface(interface, phases.vf, diagnostics, output_number++);
    }
  }

  double l1 = 0.0;
  FOR_INTERIOR(grid, i, j, k) l1 += std::abs(phases.vf(i, j, k) - initial_vf(i, j, k));
  l1 /= static_cast<double>(grid.n[0]) * grid.n[1] * grid.n[2];
  std::printf("L1 difference between start and end: %.7g\n", l1);
  std::printf("Time: total %.2f s, advection %.2f s, reconstruction %.2f s\n",
              std::chrono::duration<double>(Clock::now() - run_start).count(), advect_seconds, reconstruct_seconds);
  return 0;
}
