// Times reconstruction methods on identical inputs and compares their output.
//
//   r2p_timing CASE NCELLS DT T_SNAPSHOT REPS DRIVER METHOD [METHOD ...]
//
// Advects CASE (Film3D, Deformation3D, Sheets) with FullLagrangian and the
// DRIVER reconstruction up to T_SNAPSHOT, then reconstructs that same state
// REPS times with each METHOD (after one untimed warm-up call), restoring the
// previous-step interface before every call. Reports the minimum and median
// wall time per call, the per-stage profile of the *Fast methods, and, for
// each X / XFast pair, how far the two reconstructions are apart. The timing
// table is also appended to r2p_timing.csv (one row per case and method).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "examples/new_advector/deformation_3d.h"
#include "examples/new_advector/film_3d.h"
#include "examples/new_advector/r2p_fast.h"
#include "examples/new_advector/r2p_snap.h"
#include "examples/new_advector/reconstruction_types.h"
#include "examples/new_advector/sheets.h"
#include "examples/new_advector/solver.h"
#include "examples/new_advector/vof_advection.h"

namespace {

struct Result {
  std::string method;
  double tmin = 0.0, tmed = 0.0;
  Data<IRL::PlanarSeparator> iface;
  r2pfast::Profile profile;
};

// Largest differences between two reconstructions over the interior mixed
// cells, split into the domain-boundary layer and the rest.
void compare(const Result& a, const Result& b, const Data<double>& vf) {
  const BasicMesh& mesh = vf.getMesh();
  struct Acc {
    long cells = 0, count_diff = 0, flip_diff = 0;
    double n = 0.0, d = 0.0;
  } acc[2];
  const double h = (mesh.dx() + mesh.dy() + mesh.dz()) / 3.0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i)
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j)
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        const double f = vf(i, j, k);
        if (f < IRL::global_constants::VF_LOW || f > IRL::global_constants::VF_HIGH) continue;
        const bool edge = i == mesh.imin() || i == mesh.imax() || j == mesh.jmin() ||
                          j == mesh.jmax() || k == mesh.kmin() || k == mesh.kmax();
        Acc& s = acc[edge ? 1 : 0];
        const IRL::PlanarSeparator& p = a.iface(i, j, k);
        const IRL::PlanarSeparator& q = b.iface(i, j, k);
        ++s.cells;
        if (p.getNumberOfPlanes() != q.getNumberOfPlanes()) {
          ++s.count_diff;
          continue;
        }
        if (p.getNumberOfPlanes() == 2 && p.isFlipped() != q.isFlipped()) ++s.flip_diff;
        for (IRL::UnsignedIndex_t n = 0; n < p.getNumberOfPlanes(); ++n) {
          s.n = std::max(s.n, IRL::magnitude(p[n].normal() - q[n].normal()));
          s.d = std::max(s.d, std::abs(p[n].distance() - q[n].distance()) / h);
        }
      }
  std::printf("  %s vs %s\n", a.method.c_str(), b.method.c_str());
  const char* name[2] = {"interior      ", "boundary layer"};
  for (int e = 0; e < 2; ++e)
    std::printf("    %s %6ld mixed cells: %5ld plane-count and %ld flip differences, "
                "max |dn| %.2e, max |dd|/dx %.2e\n",
                name[e], acc[e].cells, acc[e].count_diff, acc[e].flip_diff, acc[e].n, acc[e].d);
}

// Thin films (two planes, VF <= 0.1): how many are snapped slabs, and how
// many pinch inside the cell. The film gap is linear in space, so its minimum
// over the cell is at a corner; "pinched" = minimum < half the gap at the
// centre, "crossed" = minimum < 0 (the planes meet inside the cell).
void thinFilmReport(const Result& r, const Data<double>& vf) {
  const BasicMesh& mesh = vf.getMesh();
  long n = 0, slab = 0, pinched = 0, crossed = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i)
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j)
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        const IRL::PlanarSeparator& sep = r.iface(i, j, k);
        if (sep.getNumberOfPlanes() != 2 || vf(i, j, k) > 0.1 || vf(i, j, k) < IRL::global_constants::VF_LOW) continue;
        ++n;
        slab += r2psnap::isSlab(sep);
        const double s = sep.isFlipped() ? -1.0 : 1.0;
        auto gap = [&](const IRL::Pt& p) {
          return s * ((sep[0].distance() - sep[0].normal() * p) + (sep[1].distance() - sep[1].normal() * p));
        };
        const IRL::Pt c(mesh.xm(i), mesh.ym(j), mesh.zm(k));
        double gmin = 1e30;
        for (int q = 0; q < 8; ++q)
          gmin = std::min(gmin, gap(IRL::Pt(q & 1 ? mesh.x(i + 1) : mesh.x(i), q & 2 ? mesh.y(j + 1) : mesh.y(j),
                                            q & 4 ? mesh.z(k + 1) : mesh.z(k))));
        pinched += gmin < 0.5 * gap(c);
        crossed += gmin < 0.0;
      }
  std::printf("  %-18s thin two-plane cells (VF <= 0.1): %6ld, snapped %6ld, pinched < half %5ld, planes cross %5ld\n",
              r.method.c_str(), n, slab, pinched, crossed);
}

// One row per method: timings, cell counts and the *Fast stage profile (ms).
void appendCsv(const std::string& sim, const int ncells, const double t, const long mixed,
               const int reps, const std::vector<Result>& results) {
  const char* path = "r2p_timing.csv";
  std::FILE* probe = std::fopen(path, "r");
  const bool header = probe == nullptr;
  if (probe != nullptr) std::fclose(probe);
  std::FILE* f = std::fopen(path, "a");
  if (f == nullptr) return;
  if (header) {
    std::fprintf(f, "case,ncells,time,mixed_cells,method,min_ms,median_ms,us_per_mixed_cell,"
                    "routed_to_r2p,two_plane_pass1,backprojected_sources");
    for (int s = 0; s < 12; ++s)
      if (r2pfast::kStageNames[s][0] != '\0') std::fprintf(f, ",%s_ms", r2pfast::kStageNames[s]);
    std::fprintf(f, "\n");
  }
  for (const auto& r : results) {
    const r2pfast::Profile& p = r.profile;
    std::fprintf(f, "%s,%d,%.6g,%ld,%s,%.4f,%.4f,%.3f,%ld,%ld,%ld", sim.c_str(), ncells, t, mixed,
                 r.method.c_str(), 1e3 * r.tmin, 1e3 * r.tmed, 1e6 * r.tmed / std::max(1L, mixed),
                 p.count[1] / reps, p.count[2] / reps, p.count[3] / reps);
    for (int s = 0; s < 12; ++s)
      if (r2pfast::kStageNames[s][0] != '\0') std::fprintf(f, ",%.4f", 1e3 * p.stage[s] / reps);
    std::fprintf(f, "\n");
  }
  std::fclose(f);
}

template <class SimulationType>
int run(const std::string& sim, const int ncells, const double dt, const double t_snap, const int reps,
        const std::string& driver, const std::vector<std::string>& methods) {
  BasicMesh mesh = SimulationType::setMesh(ncells);
  Data<double> U(&mesh), V(&mesh), W(&mesh), vf(&mesh);
  Data<IRL::Pt> liq(&mesh), gas(&mesh);
  Data<IRL::PlanarLocalizer> localizers(&mesh);
  initializeLocalizers(&localizers);
  Data<IRL::PlanarSeparator> iface(&mesh);
  Data<IRL::LocalizedSeparatorLink> links(&mesh);
  initializeLocalizedSeparators(localizers, iface, &links);
  connectMesh(mesh, &links);
  IRL::setMinimumVolumeToTrack(10.0 * DBL_EPSILON * mesh.dx() * mesh.dy() * mesh.dz());
  IRL::setVolumeFractionBounds(1.0e-8);
  IRL::setVolumeFractionTolerance(1.0e-13);
  SimulationType::initialize(&U, &V, &W, &iface);
  setPhaseQuantities(iface, &vf, &liq, &gas);

  // Advance to the snapshot; the last advection is left un-reconstructed.
  double t = 0.0, step = dt;
  while (true) {
    step = std::min(dt, t_snap - t);
    if (step <= 1.0e-12) step = dt;
    SimulationType::setVelocity(t + 0.5 * step, &U, &V, &W);
    advectVOF("FullLagrangian", step, U, V, W, &links, &vf, &liq, &gas);
    t += step;
    if (t >= t_snap - 1.0e-12) break;
    getReconstruction(driver, vf, liq, gas, links, step, U, V, W, &iface);
  }
  const Data<IRL::PlanarSeparator> previous = iface;
  auto restore = [&] {
    for (int i = mesh.imino(); i <= mesh.imaxo(); ++i)
      for (int j = mesh.jmino(); j <= mesh.jmaxo(); ++j)
        for (int k = mesh.kmino(); k <= mesh.kmaxo(); ++k) iface(i, j, k) = previous(i, j, k);
  };
  long mixed = 0;
  for (int i = mesh.imin(); i <= mesh.imax(); ++i)
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j)
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k)
        if (vf(i, j, k) >= IRL::global_constants::VF_LOW && vf(i, j, k) <= IRL::global_constants::VF_HIGH)
          ++mixed;
  std::printf("state: %d^3 cells, t = %.4f, dt = %.4g, %ld mixed cells\n", ncells, t, step, mixed);

  std::vector<Result> results;
  for (const auto& m : methods) {
    Result r;
    r.method = m;
    restore();
    getReconstruction(m, vf, liq, gas, links, step, U, V, W, &iface);   // warm-up
    r2pfast::g_profile.reset();
    std::vector<double> times;
    for (int rep = 0; rep < reps; ++rep) {
      restore();
      const auto t0 = std::chrono::steady_clock::now();
      getReconstruction(m, vf, liq, gas, links, step, U, V, W, &iface);
      times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    r.tmin = times.front();
    r.tmed = times[times.size() / 2];
    r.iface = iface;
    r.profile = r2pfast::g_profile;
    results.push_back(std::move(r));
  }

  std::printf("\n  %-18s %12s %12s %14s\n", "method", "min [ms]", "median [ms]", "us/mixed cell");
  for (const auto& r : results)
    std::printf("  %-18s %12.2f %12.2f %14.1f\n", r.method.c_str(), 1e3 * r.tmin, 1e3 * r.tmed,
                1e6 * r.tmed / std::max(1L, mixed));
  for (const auto& r : results) {
    const r2pfast::Profile& p = r.profile;
    if (p.count[0] == 0) continue;
    std::printf("\n  %s per call: %ld mixed, %ld routed to R2P, %ld two-plane after pass 1",
                r.method.c_str(), p.count[0] / reps, p.count[1] / reps, p.count[2] / reps);
    if (p.count[3] > 0) std::printf(", %ld back-projected source cells", p.count[3] / reps);
    std::printf("\n");
    for (int s = 0; s < 12; ++s)
      if (p.stage[s] > 0.0)
        std::printf("    %-26s %9.2f ms\n", r2pfast::kStageNames[s], 1e3 * p.stage[s] / reps);
  }
  std::printf("\n");
  appendCsv(sim, ncells, t, mixed, reps, results);
  for (const auto& r : results) thinFilmReport(r, vf);
  std::printf("\n");
  for (const auto& a : results)
    for (const auto& b : results)
      if (b.method == a.method + "Fast") compare(a, b, vf);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: r2p_timing CASE NCELLS DT T_SNAPSHOT REPS DRIVER METHOD [METHOD ...]\n");
    return 2;
  }
  const std::string sim = argv[1];
  const int ncells = std::atoi(argv[2]);
  const double dt = std::atof(argv[3]), t_snap = std::atof(argv[4]);
  const int reps = std::max(1, std::atoi(argv[5]));
  const std::string driver = argv[6];
  const std::vector<std::string> methods(argv + 7, argv + argc);
  if (sim == "Film3D") return run<Film3D>(sim, ncells, dt, t_snap, reps, driver, methods);
  if (sim == "Deformation3D") return run<Deformation3D>(sim, ncells, dt, t_snap, reps, driver, methods);
  if (sim == "Sheets") return run<Sheets>(sim, ncells, dt, t_snap, reps, driver, methods);
  std::fprintf(stderr, "unknown case %s\n", sim.c_str());
  return 2;
}
