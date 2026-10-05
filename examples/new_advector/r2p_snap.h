// Thin-film snap, shared by R2P3D_Net and R2P3D_NetFast (and their pass 2).
//
// In a film thinner than a few hundredths of a cell, R2P-Net's small error in
// the angle between its two faces (median ~0.2 deg, p99 ~1-1.5 deg) makes the
// planes pinch or cross inside the cell: the gap changes by ~opening * cell
// width, comparable to the thickness. Such films are locally parallel, so both
// normals are replaced by +-normalize(n0 - n1), a slab. Real edges and rims
// open by degrees to tens of degrees and are left alone.
//
// Thresholds (bag case, 2026-09-25; the generated sheets / wedges / edges /
// rims put the noise p99 of the opening at ~1-1.5 deg, and edges at 9+ deg):
//   thickness <= 0.05 cells   sheets pinch below half their thickness mostly
//                 at t < 0.02 cells. The thickness is estimated as film volume
//                 / film area, the area being that of the plane through the film
//                 centroid along the mean normal, clipped to the cell; unlike the
//                 raw VF it does not depend on how the film crosses the cell, and
//                 it uses the film phase (gas films included).
//   opening <= 1 deg   at the network's noise p99 in sheets; almost no edge
//                 cells (their openings are degrees to tens of degrees)
//   sheet ends (classifier class 6) are never snapped
// The Fortran port (r2p_net_tools.f90, r2p_snap_thin_film) uses the same rule
// and values.

#ifndef EXAMPLES_NEW_ADVECTOR_R2P_SNAP_H_
#define EXAMPLES_NEW_ADVECTOR_R2P_SNAP_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <Eigen/Dense>

#include "irl/generic_cutting/cut_polygon.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polygons/polygon.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"

namespace r2psnap {

// Off by default (thin films have not been seen to break without it);
// R2P_SNAP_ENABLE=1 turns it on for a run. R2P_SNAP_THICKNESS (cells) and
// R2P_SNAP_OPENING_DEG override the thresholds. The Fortran switch is
// r2p_snap_enabled in r2p_net_tools.f90.
inline double envOr(const char* name, const double fallback) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::strtod(v, nullptr) : fallback;
}
inline bool enabled() {
  static const bool v = envOr("R2P_SNAP_ENABLE", 0.0) != 0.0;
  return v;
}
inline double maxThickness() {   // cells
  static const double v = envOr("R2P_SNAP_THICKNESS", 0.05);
  return v;
}
inline double maxOpening() {   // rad; 0 = parallel faces
  static const double v = envOr("R2P_SNAP_OPENING_DEG", 1.0) * M_PI / 180.0;
  return v;
}

// Classifier id of a sheet end (ml_classifier get_class): never snapped, its
// faces really converge.
inline constexpr int kSheetEndClass = 6;

// Film thickness in cell widths: film volume over the area of the plane through
// the film centroid with normal n, clipped to the cell. Infinite if that plane
// misses the cell.
inline double filmThickness(const IRL::RectangularCuboid& cell, const double film_vf,
                            const IRL::Pt& film_centroid, const IRL::Normal& n) {
  const IRL::PlanarSeparator mid = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, n * film_centroid));
  const double area =
      std::abs(IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(cell, mid, mid[0]).calculateVolume());
  const double vol = cell.calculateVolume();
  if (area <= 1.0e-12 * std::pow(vol, 2.0 / 3.0)) return HUGE_VAL;
  return film_vf * vol / area / std::cbrt(vol);
}

// n0, n1: the two unit face normals (either sign convention, faces opposed);
// film_vf, film_centroid: volume fraction and centroid of the film phase (the
// gas when the film is gas); interface_class: the cell's classifier id.
// Returns true and makes the normals an exact slab when the cell is not a
// sheet end, the opening is at noise level, and the film is thin.
inline bool snapThinFilm(const IRL::RectangularCuboid& cell, const double film_vf,
                         const IRL::Pt& film_centroid, const int interface_class,
                         IRL::Normal& n0, IRL::Normal& n1) {
  if (!enabled() || interface_class == kSheetEndClass) return false;
  const double opening = std::acos(std::max(-1.0, std::min(1.0, -IRL::dotProduct(n0, n1))));
  if (opening > maxOpening()) return false;
  IRL::Normal avg = n0 - n1;
  avg.normalize();
  if (filmThickness(cell, film_vf, film_centroid, avg) > maxThickness()) return false;
  n0 = avg;
  n1 = -avg;
  return true;
}

// Two exactly opposite normals: a snapped slab (the distance solve and border
// corrections change only distances, and no network output is exactly
// antiparallel). Pass 2 keeps these cells slabs.
inline bool isSlab(const IRL::PlanarSeparator& sep) {
  if (sep.getNumberOfPlanes() != 2) return false;
  const IRL::Normal& a = sep[0].normal();
  const IRL::Normal& b = sep[1].normal();
  return a[0] == -b[0] && a[1] == -b[1] && a[2] == -b[2];
}

// Very thin film: PCA slab. Far below the training range (bag films of ~1e-4
// cells) R2P-Net's normals become unreliable and holes form. Where the film in
// the 3^3 stencil is at most pcaSlabThickness() cells thick (stencilThickness
// along the PCA direction), both normals become +-the PCA direction of the 3^3
// film-phase centroids (the network's own input direction; on a thin film
// those centroids lie on its mid-surface): a slab. No guard condition: at this
// thickness a slab differs negligibly from the real faces even at a film's
// end. Default 0.005 cells, the thinnest films in the training data;
// R2P_PCA_SLAB_THICKNESS (cells) overrides it, 0 turns it off. Fortran:
// r2p_pca_slab in r2p_net_tools.f90.
inline double pcaSlabThickness() {
  static const double v = envOr("R2P_PCA_SLAB_THICKNESS", 0.005);
  return v;
}

// Film thickness over the 3^3 stencil around (i,j,k), in cell widths: the film
// phase's volume in the stencil over the area of the plane with normal n
// through its centroid, clipped to the stencil. Unlike filmThickness of the
// cell alone, it does not read thin where the cell only clips a thicker film.
// Infinite if there is no film or the plane misses the stencil. Fortran:
// r2p_stencil_thickness.
inline double stencilThickness(const BasicMesh& mesh, const Data<double>& vf, const Data<IRL::Pt>& liq,
                               const Data<IRL::Pt>& gas, const int i, const int j, const int k,
                               const bool film_is_gas, const IRL::Normal& n) {
  const double cell_vol = mesh.dx() * mesh.dy() * mesh.dz();
  double volume = 0.0, moment[3] = {0.0, 0.0, 0.0};
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) {
        const double f = film_is_gas ? 1.0 - vf(ii, jj, kk) : vf(ii, jj, kk);
        if (!(f > 0.0)) continue;
        const IRL::Pt& c = film_is_gas ? gas(ii, jj, kk) : liq(ii, jj, kk);
        volume += f * cell_vol;
        for (int a = 0; a < 3; ++a) moment[a] += f * cell_vol * c[a];
      }
  if (!(volume > 0.0)) return HUGE_VAL;
  const IRL::Pt centroid(moment[0] / volume, moment[1] / volume, moment[2] / volume);
  const IRL::RectangularCuboid block = IRL::RectangularCuboid::fromBoundingPts(
      IRL::Pt(mesh.x(i - 1), mesh.y(j - 1), mesh.z(k - 1)), IRL::Pt(mesh.x(i + 2), mesh.y(j + 2), mesh.z(k + 2)));
  const IRL::PlanarSeparator mid = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(n, n * centroid));
  const double area =
      std::abs(IRL::getPlanePolygonFromReconstruction<IRL::Polygon>(block, mid, mid[0]).calculateVolume());
  if (area <= 1.0e-12 * std::pow(cell_vol, 2.0 / 3.0)) return HUGE_VAL;
  return volume / area / std::cbrt(cell_vol);
}

// pca: the PCA direction (physical, unit). On success n0, n1 are +-pca in the
// network's (cell-index) frame, as its own normals are before mesh scaling.
inline bool pcaSlab(const BasicMesh& mesh, const Data<double>& vf, const Data<IRL::Pt>& liq,
                    const Data<IRL::Pt>& gas, const int i, const int j, const int k, const bool film_is_gas,
                    const IRL::Normal& pca, IRL::Normal& n0, IRL::Normal& n1) {
  if (!(stencilThickness(mesh, vf, liq, gas, i, j, k, film_is_gas, pca) <= pcaSlabThickness())) return false;
  n0 = IRL::Normal(pca[0] / mesh.dx(), pca[1] / mesh.dy(), pca[2] / mesh.dz());
  n0.normalize();
  n1 = -n0;
  return true;
}

// PCA direction of the film-phase centroids of the 3^3 cells with film VF >
// VF_LOW, as the R2P-Net input computes it (R2P3D_Net PCA_Normal, R2P3D_NetFast
// pcaNormal); false with fewer than 6 such cells.
inline bool stencilPca(const Data<double>& vf, const Data<IRL::Pt>& liq, const Data<IRL::Pt>& gas, const int i,
                       const int j, const int k, const bool film_is_gas, IRL::Normal* pca) {
  Eigen::Vector3d pts[27];
  int np = 0;
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) {
        const double f = film_is_gas ? 1.0 - vf(ii, jj, kk) : vf(ii, jj, kk);
        if (!(f > IRL::global_constants::VF_LOW)) continue;
        const IRL::Pt& c = film_is_gas ? gas(ii, jj, kk) : liq(ii, jj, kk);
        pts[np++] = Eigen::Vector3d(c[0], c[1], c[2]);
      }
  if (np < 6) return false;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (int q = 0; q < np; ++q) centroid += pts[q];
  centroid = centroid / double(np);
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (int q = 0; q < np; ++q) covariance += (pts[q] - centroid) * (pts[q] - centroid).transpose();
  const Eigen::Vector3d n = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(covariance).eigenvectors().col(0).normalized();
  *pca = IRL::Normal(n[0], n[1], n[2]);
  pca->normalize();
  return true;
}

// Very thin film, for the routing: the cells pcaSlab turns into slabs (at least
// 6 film cells in the 3^3 stencil, stencilThickness along their PCA direction
// at most pcaSlabThickness()). They go to R2P-Net whatever the classifier,
// detector or guard says, so they get the slab rather than PLICNet's single
// plane. A quick bound skips the PCA for thicker films: the 3^3 block's largest
// cross-section is 3^2 sqrt(2) < 16 cell faces, so a film of thickness t holds
// less than 16 t cell volumes there. Fortran: r2p_very_thin_stencil.
inline bool veryThinStencil(const BasicMesh& mesh, const Data<double>& vf, const Data<IRL::Pt>& liq,
                            const Data<IRL::Pt>& gas, const int i, const int j, const int k, const bool film_is_gas) {
  const double t_max = pcaSlabThickness();
  if (!(t_max > 0.0)) return false;
  double volume = 0.0;
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) volume += film_is_gas ? 1.0 - vf(ii, jj, kk) : vf(ii, jj, kk);
  if (volume > 16.0 * t_max) return false;
  IRL::Normal pca;
  if (!stencilPca(vf, liq, gas, i, j, k, film_is_gas, &pca)) return false;
  return stencilThickness(mesh, vf, liq, gas, i, j, k, film_is_gas, pca) <= t_max;
}

// Thin-film guard slab. Where the thin-film guard holds (r2p_edge_topology.h
// filmSeparates: the other phase lies on both sides of the film, apart), the
// film runs through the cell and needs two planes; but far below its training
// range (bag films of ~1e-4 cells) R2P-Net can shrink one normal below 0.85.
// The cell then gets a parallel slab around the longer network normal instead
// of one plane. False (one plane stays) if both normals are zero. Fortran:
// r2p_guard_slab in r2p_net_tools.f90.
inline bool guardSlab(IRL::Normal& n0, IRL::Normal& n1) {
  const double m0 = n0.calculateMagnitude(), m1 = n1.calculateMagnitude();
  if (std::max(m0, m1) <= 0.0) return false;
  if (m0 >= m1)
    n1 = -n0;
  else
    n0 = -n1;
  return true;
}

// Pass 2's fitted normals for a slab cell: rotate the slab, keep it parallel.
inline void keepSlab(IRL::Normal& n0, IRL::Normal& n1) {
  IRL::Normal avg = n0 - n1;
  avg.normalize();
  n0 = avg;
  n1 = -avg;
}

}  // namespace r2psnap

#endif  // EXAMPLES_NEW_ADVECTOR_R2P_SNAP_H_
