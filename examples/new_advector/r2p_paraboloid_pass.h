// #ifndef R2P_PARABOLOID_PASS_H_
// #define R2P_PARABOLOID_PASS_H_

// #include <algorithm>
// #include <cmath>
// #include <vector>

// #include "irl/geometry/general/pt.h"
// #include "irl/geometry/polyhedrons/rectangular_cuboid.h"
// #include "irl/generic_cutting/generic_cutting.h"
// #include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
// #include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
// #include "irl/moments/separated_volume_moments.h"
// #include "irl/moments/volume_moments.h"
// #include "irl/parameters/constants.h"
// #include "irl/planar_reconstruction/planar_separator.h"

// #include "examples/new_advector/basic_mesh.h"
// #include "examples/new_advector/data.h"
// #include "examples/new_advector/r2p_refine.h"
// #include "examples/new_advector/r2p_refine_coupled.h"

// // Second-pass paraboloid refinement.
// //
// // Pass 1 (the existing R2P3D_Net loop) fills a_interface with ML normals at
// // volume-conserving distances. This pass then sweeps the whole field: for
// // each mixed two-plane cell it fits paraboloids to the neighborhood's
// // reconstructed polygons, replaces the normals, and re-solves distance.
// //
// // WHY A SEPARATE PASS. Refining inline reads neighbors from an a_interface
// // that is half-updated -- cells already visited hold new reconstructions,
// // the rest hold last timestep's. That asymmetry follows the sweep direction
// // and shows up as directional bias in a translation test. Here the field is
// // snapshotted first, so every cell fits against the same data and the result
// // is independent of visit order.
// //
// // WHY R2PDistanceSolver AND NOT ...Solver2. Solver2 runs a Levenberg-
// // Marquardt search over tilt, splay, and offset: handed a paraboloid normal
// // it would immediately rotate away from it, and the pass would measure the
// // LM's fixed point rather than the paraboloid's answer. R2PDistanceSolver
// // holds both normals fixed and bisects a single common shift, which is
// // exactly the remaining degree of freedom once orientation is settled.

// // R2PDistanceSolver is defined in reconstruction_types.cpp with no
// // declaration in any header. Declared here so this pass can call it; move it
// // into reconstruction_types.h if you would rather not duplicate the
// // signature. It takes the cell BY VALUE, matching the existing definition.
// void R2PDistanceSolver(double VF_target, IRL::Pt bary_target,
//                        IRL::PlanarSeparator& a_interface,
//                        IRL::RectangularCuboid cell);

// namespace r2ppass {

// struct Options {
//   r2pfit::CoupledOptions coupled;   // used when use_coupled_fit = true
//   r2pfit::Options uncoupled;        // used when use_coupled_fit = false
//   bool use_coupled_fit = true;
//   // Coupled is the default. The uncoupled path is kept switchable for A/B
//   // comparison, not because it is recommended -- it is what produces the
//   // spurious planes in flat regions.

//   bool refine_two_plane = true;

//   int sweeps = 1;
//   // ITERATION. The fit reads reconstructed PLIC polygons, so its input error
//   // is pass 1's normal error. Re-running the whole pass feeds it its own
//   // improved output: for a flat sheet this contracts toward the fixed point
//   // where every polygon centroid is coplanar, which is the exact plane. One
//   // sweep leaves pass-1 error baked in; 2-3 sweeps recovers most of what the
//   // LM got for free from its moment-based objective. Each sweep re-snapshots,
//   // so ordering stays symmetric. Cost is linear in sweeps.

//   bool select_plane_count = true;
//   double plane_drop_bias = 1.05;
//   // PLANE-COUNT SELECTION. cleanReconstruction only drops a plane that fails
//   // to cut the cell, so a slightly-wrong normal keeps a spurious second plane
//   // alive. This instead compares the two-plane reconstruction against the
//   // best single-plane one on centroid error and keeps whichever is better,
//   // with a bias > 1 favouring the simpler model on ties. This is a model
//   // selection step, not a geometric cleanup, and it is what replaces the
//   // LM's ability to push a plane out of the cell.
// };

// struct Stats {
//   int visited = 0;
//   int refined = 0;
//   int split_failed = 0;
//   int fit_rejected = 0;
//   int splay_rejected = 0;   // only incremented on the coupled path
//   int collapsed_to_one = 0;
//   int snapped_planar = 0;
// };

// // Assembles the 3x3x3 CellPlanes stencil for cell (i,j,k) from `snapshot`.
// // Center goes in FIRST: the fit takes polys[0] as the reference point and
// // frame, so the fitted normal is the surface normal at this cell's own
// // interface rather than at some arbitrary neighbor's.
// inline std::vector<r2pfit::CellPlanes> buildStencil(
//     const BasicMesh& mesh, const Data<double>& a_liquid_volume_fraction,
//     const Data<IRL::PlanarSeparator>& snapshot, const int i, const int j,
//     const int k) {
//   std::vector<r2pfit::CellPlanes> cells;
//   cells.reserve(27);

//   for (int pass = 0; pass < 2; ++pass) {
//     for (int ii = i - 1; ii <= i + 1; ++ii) {
//       for (int jj = j - 1; jj <= j + 1; ++jj) {
//         for (int kk = k - 1; kk <= k + 1; ++kk) {
//           const bool is_center = (ii == i && jj == j && kk == k);
//           // pass 0 takes the center only; pass 1 takes everything else.
//           if ((pass == 0) != is_center) continue;

//           r2pfit::CellPlanes cp;
//           cp.lo = IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk));
//           cp.hi = IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1));
//           cp.separator = snapshot(ii, jj, kk);
//           const double vf = a_liquid_volume_fraction(ii, jj, kk);
//           cp.mixed = vf > IRL::global_constants::VF_LOW &&
//                      vf < IRL::global_constants::VF_HIGH &&
//                      cp.separator.getNumberOfPlanes() > 0;
//           cells.push_back(cp);
//         }
//       }
//     }
//   }
//   return cells;
// }

// // Distance from a reconstruction's liquid centroid to the target centroid.
// // Volume fraction is matched exactly by construction in both candidates, so
// // the centroid is the only discriminating moment left -- which makes this the
// // same quantity the LM was minimizing, just evaluated rather than searched.
// //
// // NOTE: confirm this getNormalizedVolumeMoments spelling against the calls in
// // reconstruction_types.cpp; the template argument list is the fragile part.
// inline double centroidError(const IRL::RectangularCuboid& cell,
//                             const IRL::PlanarSeparator& sep,
//                             const IRL::Pt& target_centroid) {
//   const IRL::SeparatedMoments<IRL::VolumeMoments> moments =
//       IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>,
//                                       IRL::ReconstructionDefaultCuttingMethod>(
//           cell, sep);
//   const IRL::Pt c = moments[0].centroid();
//   const double dx = c[0] - target_centroid[0];
//   const double dy = c[1] - target_centroid[1];
//   const double dz = c[2] - target_centroid[2];
//   return std::sqrt(dx * dx + dy * dy + dz * dz);
// }

// // Refines every two-plane cell in the field. a_liquid_centroid is passed
// // straight to R2PDistanceSolver, which converts it internally when the
// // separator is flipped -- always hand it the LIQUID centroid regardless of
// // flip state.
// //
// // a_branch is templated because `branch` in reconstruction_types.cpp may be
// // Data<int> or Data<double>; pass nullptr to skip the diagnostic entirely.
// template <class BranchDataType>
// inline Stats runSweep(const Data<double>& a_liquid_volume_fraction,
//                       const Data<IRL::Pt>& a_liquid_centroid,
//                       Data<IRL::PlanarSeparator>* a_interface,
//                       BranchDataType* a_branch, const Options& options) {
//   const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
//   Stats stats;

//   // Snapshot BEFORE any refinement so neighbor data is uniform. Data is
//   // copyable (solver.h relies on this for the L1 diagnostic).
//   const Data<IRL::PlanarSeparator> snapshot = *a_interface;

//   const double cell_width = (mesh.dx() + mesh.dy() + mesh.dz()) / 3.0;

//   r2pfit::CoupledOptions coupled_options = options.coupled;
//   if (coupled_options.fit.mesh_size == 1.0) {
//     coupled_options.fit.mesh_size = cell_width;
//   }
//   r2pfit::Options uncoupled_options = options.uncoupled;
//   if (uncoupled_options.mesh_size == 1.0) {
//     uncoupled_options.mesh_size = cell_width;
//   }

//   if (!options.refine_two_plane) return stats;

//   for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
//     for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
//       for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
//         const double vf = a_liquid_volume_fraction(i, j, k);
//         if (vf <= IRL::global_constants::VF_LOW ||
//             vf >= IRL::global_constants::VF_HIGH) {
//           continue;
//         }

//         IRL::PlanarSeparator& sep = (*a_interface)(i, j, k);
//         if (sep.getNumberOfPlanes() != 2) continue;

//         ++stats.visited;

//         IRL::Normal normal1 = sep[0].normal();
//         IRL::Normal normal2 = sep[1].normal();
//         normal1.normalize();
//         normal2.normalize();

//         const std::vector<r2pfit::CellPlanes> cells = buildStencil(
//             mesh, a_liquid_volume_fraction, snapshot, i, j, k);

//         bool updated = false;
//         if (options.use_coupled_fit) {
//           r2pfit::CoupledResult fit_result;
//           updated = r2pfit::refineTwoNormalsCoupled(
//               cells, normal1, normal2, &fit_result, coupled_options);
//           if (!updated) {
//             if (fit_result.split_failed) {
//               ++stats.split_failed;
//             } else if (fit_result.splay_rejected) {
//               ++stats.splay_rejected;
//             } else {
//               ++stats.fit_rejected;
//             }
//             continue;   // leave pass-1 reconstruction untouched
//           }
//         } else {
//           r2pfit::Result fit_result;
//           updated = r2pfit::refineTwoNormals(cells, normal1, normal2,
//                                              &fit_result, uncoupled_options);
//           if (!updated) {
//             if (fit_result.split_failed) {
//               ++stats.split_failed;
//             } else {
//               ++stats.fit_rejected;
//             }
//             continue;
//           }
//         }
//         ++stats.refined;

//         // Rebuild with the fitted normals, preserving pass 1's flip state,
//         // then translate to conserve volume. Distances start at zero because
//         // R2PDistanceSolver derives its own initial pair from the bisector
//         // projection of the target centroid.
//         const double flip_i = sep.isNotFlipped() ? 1.0 : -1.0;
//         const IRL::RectangularCuboid cube =
//             IRL::RectangularCuboid::fromBoundingPts(
//                 IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
//                 IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));

//         sep = IRL::PlanarSeparator::fromTwoPlanes(
//             IRL::Plane(normal1, 0.0), IRL::Plane(normal2, 0.0), flip_i);
//         R2PDistanceSolver(vf, a_liquid_centroid(i, j, k), sep, cube);

//         // Model selection: is one plane actually better here?
//         if (options.select_plane_count && sep.getNumberOfPlanes() == 2) {
//           const double err_two =
//               centroidError(cube, sep, a_liquid_centroid(i, j, k));

//           // Candidate single planes: each fitted normal on its own. The
//           // dominant surface is not always the larger-area group once the
//           // second plane is nearly out of the cell, so both are tried.
//           double err_one = -1.0;
//           IRL::PlanarSeparator best_one;
//           for (int g = 0; g < 2; ++g) {
//             IRL::PlanarSeparator cand = IRL::PlanarSeparator::fromOnePlane(
//                 IRL::Plane((g == 0) ? normal1 : normal2, 0.0));
//             IRL::setDistanceToMatchVolumeFraction(cube, vf, &cand);
//             const double e = centroidError(cube, cand, a_liquid_centroid(i, j, k));
//             if (err_one < 0.0 || e < err_one) {
//               err_one = e;
//               best_one = cand;
//             }
//           }

//           if (err_one >= 0.0 && err_one <= options.plane_drop_bias * err_two) {
//             sep = best_one;
//             ++stats.collapsed_to_one;
//           }
//         }

//         if (a_branch != nullptr) (*a_branch)(i, j, k) = 6;
//       }
//     }
//   }

//   a_interface->updateBorder();
//   return stats;
// }

// // Runs `options.sweeps` sweeps. Each one re-snapshots the field, so the fit
// // always sees a consistent neighborhood, and each feeds on the previous
// // sweep's improved normals. Reported stats are from the LAST sweep, which is
// // the one describing the final state; earlier sweeps typically show more
// // refinements and fewer rejections as the field settles.
// template <class BranchDataType>
// inline Stats runImpl(const Data<double>& a_liquid_volume_fraction,
//                      const Data<IRL::Pt>& a_liquid_centroid,
//                      Data<IRL::PlanarSeparator>* a_interface,
//                      BranchDataType* a_branch, const Options& options) {
//   Stats stats;
//   const int nsweeps = std::max(1, options.sweeps);
//   for (int sweep = 0; sweep < nsweeps; ++sweep) {
//     stats = runSweep(a_liquid_volume_fraction, a_liquid_centroid, a_interface,
//                      a_branch, options);
//   }
//   return stats;
// }

// // Convenience overloads.
// template <class BranchDataType>
// inline Stats run(const Data<double>& a_liquid_volume_fraction,
//                  const Data<IRL::Pt>& a_liquid_centroid,
//                  Data<IRL::PlanarSeparator>* a_interface,
//                  BranchDataType* a_branch,
//                  const Options& options = Options()) {
//   return runImpl(a_liquid_volume_fraction, a_liquid_centroid, a_interface,
//                  a_branch, options);
// }

// inline Stats run(const Data<double>& a_liquid_volume_fraction,
//                  const Data<IRL::Pt>& a_liquid_centroid,
//                  Data<IRL::PlanarSeparator>* a_interface,
//                  const Options& options = Options()) {
//   return runImpl<Data<int>>(a_liquid_volume_fraction, a_liquid_centroid,
//                             a_interface, nullptr, options);
// }

// }  // namespace r2ppass

// #endif  // R2P_PARABOLOID_PASS_H_



#ifndef R2P_PARABOLOID_PASS_H_
#define R2P_PARABOLOID_PASS_H_

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/moments/separated_volume_moments.h"
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "irl/machine_learning_reconstruction/plic_geometry.h"
#include "irl/machine_learning_reconstruction/plic_paraboloid.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"
#include "examples/new_advector/r2p_newton_distance.h"
#include "examples/new_advector/r2p_snap.h"
#include "examples/new_advector/r2p_nopinch.h"

// ===========================================================================
// Paraboloid refinement for R2P reconstructions (pass 2 of R2P3D_Net).
//
// For each two-plane cell: collect the 3^3 stencil's interface polygons, sort
// them into the film's two faces by orientation, fit both faces jointly (one
// frame, shared curvature, separate offsets -- fitting them independently lets
// the wedge angle drift), replace the normals, re-solve the distances, and
// choose between one and two planes.
//
//   r2pgeom    polygon extraction from a PlanarSeparator
//   r2psort    sorting polygons into the two faces
//   r2pcouple  the coupled two-face paraboloid fit
//   r2ppass    the field sweep
// ===========================================================================

// Legacy distance solve, defined in reconstruction_types.cpp (kept for the
// commented-out comparison call in runImpl).
void R2PDistanceSolver(double VF_target, IRL::Pt bary_target,
                       IRL::PlanarSeparator& a_interface,
                       IRL::RectangularCuboid cell);

// ===========================================================================
namespace r2pgeom {
// ===========================================================================

// One cell's geometry and its current reconstruction. Corner points rather
// than a RectangularCuboid because plicgeom::planeBoxPolygon takes corners.
struct CellPlanes {
  IRL::Pt lo;
  IRL::Pt hi;
  IRL::PlanarSeparator separator;
  bool mixed = false;
};

// One plane of one cell, tagged with the surface group it belongs to.
struct Tagged {
  plicparab::SurfacePolygon poly;
  int group = -1;
  bool is_center = false;
};

// Rotates `from` toward `to` by at most `max_angle` radians (Rodrigues).
inline IRL::Normal limitedRotation(const IRL::Normal& from,
                                   const IRL::Normal& to,
                                   const double max_angle, double* achieved) {
  const double cos_angle = std::max(-1.0, std::min(1.0, from * to));
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
// triangulation from vertex 0. False on a degenerate polygon.
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

// Sutherland-Hodgman clip of a convex planar polygon to {x : n.x - d <= 0}.
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

// One plane of one cell's separator as a SurfacePolygon: the box polygon,
// clipped to the part that actually bounds the phases. Unflipped, the liquid
// is the intersection of the planes' below-sides, so plane p's interface lies
// below every other plane; flipped, the liquid is their union, so it lies
// ABOVE every other plane.
inline std::optional<plicparab::SurfacePolygon> polygonFor(
    const IRL::Pt& lo, const IRL::Pt& hi, const IRL::PlanarSeparator& sep,
    const IRL::UnsignedIndex_t p) {
  const auto info =
      plicgeom::planeBoxPolygon(lo, hi, sep[p].normal(), sep[p].distance());
  if (!info) return std::nullopt;

  const double side = sep.isFlipped() ? -1.0 : 1.0;
  std::vector<IRL::Pt> verts = info->vertices;
  for (IRL::UnsignedIndex_t q = 0; q < sep.getNumberOfPlanes(); ++q) {
    if (q == p) continue;
    verts = clipToHalfSpace(verts, side * sep[q].normal(), side * sep[q].distance());
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

// Flattens every plane in the stencil into Tagged polygons, recording where
// each cell's run starts so the sort can pair a cell's two planes.
inline void flattenStencil(const std::vector<CellPlanes>& cells,
                           std::vector<Tagged>* tagged,
                           std::vector<std::size_t>* cell_begin) {
  tagged->clear();
  cell_begin->clear();
  cell_begin->push_back(0);
  for (std::size_t c = 0; c < cells.size(); ++c) {
    if (cells[c].mixed) {
      const IRL::PlanarSeparator& sep = cells[c].separator;
      for (IRL::UnsignedIndex_t p = 0; p < sep.getNumberOfPlanes(); ++p) {
        std::optional<plicparab::SurfacePolygon> sp =
            polygonFor(cells[c].lo, cells[c].hi, sep, p);
        if (!sp) continue;
        Tagged t;
        t.poly = *sp;
        t.is_center = (c == 0);
        tagged->push_back(t);
      }
    }
    cell_begin->push_back(tagged->size());
  }
}

}  // namespace r2pgeom

// ===========================================================================
namespace r2psort {
// ===========================================================================

struct Options {
  double max_rotation = 0.35;             // rad; fits beyond this are clamped
  double min_group_area_fraction = 0.10;  // smaller group share: keep network normals
  double min_split_dot = 0.0;             // min dot with the assigned group normal
};

// Assigns each polygon to the face whose network normal it best matches.
// Orientation only: the faces are under a cell apart but nearly antiparallel.
// A cell's two planes are assigned jointly (best total dot product), so both
// cannot land on the same face.
inline void sortPlanes(std::vector<r2pgeom::Tagged>& tagged,
                       const std::vector<std::size_t>& cell_begin,
                       const IRL::Normal& n0, const IRL::Normal& n1,
                       const Options& opt) {
  for (std::size_t c = 0; c + 1 < cell_begin.size(); ++c) {
    const std::size_t begin = cell_begin[c];
    const std::size_t end = cell_begin[c + 1];
    const std::size_t count = end - begin;

    if (count == 1) {
      r2pgeom::Tagged& t = tagged[begin];
      const double d0 = t.poly.normal * n0;
      const double d1 = t.poly.normal * n1;
      const int g = (d0 >= d1) ? 0 : 1;
      t.group = (std::max(d0, d1) > opt.min_split_dot) ? g : -1;
    } else if (count >= 2) {
      r2pgeom::Tagged& a = tagged[begin];
      r2pgeom::Tagged& b = tagged[begin + 1];
      const double a0 = a.poly.normal * n0;
      const double a1 = a.poly.normal * n1;
      const double b0 = b.poly.normal * n0;
      const double b1 = b.poly.normal * n1;
      if (a0 + b1 >= a1 + b0) {          // pairing a->0, b->1
        a.group = (a0 > opt.min_split_dot) ? 0 : -1;
        b.group = (b1 > opt.min_split_dot) ? 1 : -1;
      } else {                            // pairing a->1, b->0
        a.group = (a1 > opt.min_split_dot) ? 1 : -1;
        b.group = (b0 > opt.min_split_dot) ? 0 : -1;
      }
      // Any further planes in this cell (rare) take the one-plane rule.
      for (std::size_t extra = begin + 2; extra < end; ++extra) {
        r2pgeom::Tagged& t = tagged[extra];
        const double d0 = t.poly.normal * n0;
        const double d1 = t.poly.normal * n1;
        const int g = (d0 >= d1) ? 0 : 1;
        t.group = (std::max(d0, d1) > opt.min_split_dot) ? g : -1;
      }
    }
  }
}

// Area share held by each group. False if there is no area at all.
inline bool groupAreaFractions(const std::vector<r2pgeom::Tagged>& tagged,
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

// One group's polygons, centre-cell polygon first (the fit's reference point).
// Empty if the group has no centre polygon.
inline std::vector<plicparab::SurfacePolygon> gatherGroup(
    const std::vector<r2pgeom::Tagged>& tagged, const int group) {
  std::vector<plicparab::SurfacePolygon> polys;
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    if (tagged[i].group == group && tagged[i].is_center) {
      polys.push_back(tagged[i].poly);
    }
  }
  if (polys.empty()) return polys;
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    if (tagged[i].group == group && !tagged[i].is_center) {
      polys.push_back(tagged[i].poly);
    }
  }
  return polys;
}

}  // namespace r2psort

// ===========================================================================
namespace r2pcouple {
// ===========================================================================
//
// Both faces as height fields over one (t,s) tangent plane:
//
//   face g:  n = a0_g + (a1+d1_g) t + (a2+d2_g) s + a3 t^2 + a4 ts + a5 s^2
//
// a1..a5 are shared (common shape), a0_g are the offsets (their difference is
// the thickness), d1_g/d2_g are per-face splay with a ridge penalty. Only
// a1+d1_g is identifiable; the SVD's minimum-norm solution splits it evenly.

struct Options {
  double h = 2.5;                    // Gaussian weight radius, in mesh_size units
  double mesh_size = 1.0;            // set to the cell width by r2ppass::run
  double splay_penalty = 0.015;      // ridge on d*_g: large forces a parallel slab
  double curvature_penalty = 1.0e-3; // ridge on a3..a5: tames near-collinear samples
  int min_per_group = 6;
  double max_resid = 0.25;

  // Tangent-plane frame: group 0's centre polygon, or the label-symmetric
  // bisector n0 - n1 (unweighted, or weighted by centre-polygon area).
  enum class FrameSource { kGroup0, kBisector, kAreaWeighted };
  FrameSource frame_source = FrameSource::kBisector;
  bool bisector_origin = true;           // origin at the midpoint of the centre centroids
  double min_bisector_magnitude = 1.0e-3; // below: faces parallel, fall back to kGroup0
};

struct Result {
  IRL::Normal normal[2];
  double curvature[2] = {0.0, 0.0};
  double thickness = 0.0;      // |a0_1 - a0_0| * mesh_size
  double splay_angle = 0.0;    // radians between the two fitted normals
  double rms_residual = 0.0;
  int count[2] = {0, 0};
};

namespace detail {
// [a0_0, a0_1, a1, a2, a3, a4, a5, d1_0, d2_0, d1_1, d2_1]
constexpr int kNumUnknowns = 11;
// 4 splay rows + 3 curvature rows.
constexpr int kNumPenaltyRows = 7;
using CoeffVector = Eigen::Matrix<double, kNumUnknowns, 1>;
}  // namespace detail

// polys0/polys1 must each have that group's CENTER-CELL polygon at index 0.
// seed0/seed1 are the R2PNet normals, used only to resolve sign.
inline std::optional<Result> fitCoupled(
    const std::vector<plicparab::SurfacePolygon>& polys0,
    const std::vector<plicparab::SurfacePolygon>& polys1,
    const IRL::Normal& seed0, const IRL::Normal& seed1,
    const Options& opt = Options()) {
  if (static_cast<int>(polys0.size()) < opt.min_per_group ||
      static_cast<int>(polys1.size()) < opt.min_per_group) {
    return std::nullopt;
  }

  // One frame for both faces. Group 1 faces roughly -nref, so the symmetric
  // direction is n0 - n1; the sign is restored at extraction.
  IRL::Pt pref = polys0[0].centroid;
  IRL::Normal seed_normal = polys0[0].normal;

  if (opt.frame_source != Options::FrameSource::kGroup0) {
    IRL::Normal n0 = polys0[0].normal;
    IRL::Normal n1 = polys1[0].normal;
    n0.normalize();
    n1.normalize();

    double w0 = 0.5;
    double w1 = 0.5;
    if (opt.frame_source == Options::FrameSource::kAreaWeighted) {
      const double wsum = polys0[0].area + polys1[0].area;
      if (wsum > 0.0) {
        w0 = polys0[0].area / wsum;
        w1 = polys1[0].area / wsum;
      }
    }

    const IRL::Normal combined = w0 * n0 - w1 * n1;
    if (combined.calculateMagnitude() >= opt.min_bisector_magnitude) {
      seed_normal = combined;
      if (opt.bisector_origin) {
        pref = IRL::Pt(w0 * polys0[0].centroid[0] + w1 * polys1[0].centroid[0],
                       w0 * polys0[0].centroid[1] + w1 * polys1[0].centroid[1],
                       w0 * polys0[0].centroid[2] + w1 * polys1[0].centroid[2]);
      }
    }
  }

  IRL::Normal nref, tref, sref;
  plicparab::buildFrame(seed_normal, &nref, &tref, &sref);

  std::vector<detail::CoeffVector> rows;
  std::vector<double> rhs;
  double weight_total = 0.0;
  int count[2] = {0, 0};

  for (int g = 0; g < 2; ++g) {
    const std::vector<plicparab::SurfacePolygon>& polys =
        (g == 0) ? polys0 : polys1;
    // Group 1 faces away from nref, so its alignment is taken against -nref.
    const IRL::Normal facing = (g == 0) ? nref : -nref;

    for (std::size_t idx = 0; idx < polys.size(); ++idx) {
      const plicparab::SurfacePolygon& sp = polys[idx];
      IRL::Normal nglob = sp.normal;
      nglob.normalize();

      const IRL::Pt dc((sp.centroid[0] - pref[0]) / opt.mesh_size,
                       (sp.centroid[1] - pref[1]) / opt.mesh_size,
                       (sp.centroid[2] - pref[2]) / opt.mesh_size);
      const double pt = tref * dc;
      const double ps = sref * dc;
      const double pn = nref * dc;
      const double dist = std::sqrt(pt * pt + ps * ps + pn * pn);
      const double wg = plicparab::wgauss(dist, opt.h);
      if (wg <= 0.0) continue;

      const double align = std::max(nglob * facing, 0.0);
      const double surf = sp.area / (opt.mesh_size * opt.mesh_size);
      const double w = surf * align * wg;
      if (w <= 0.0) continue;

      const double sw = std::sqrt(w);
      detail::CoeffVector row = detail::CoeffVector::Zero();
      row(g) = sw;                       // a0_0 or a0_1
      row(2) = sw * pt;                  // a1  (shared)
      row(3) = sw * ps;                  // a2  (shared)
      row(4) = sw * pt * pt;             // a3  (shared)
      row(5) = sw * pt * ps;             // a4  (shared)
      row(6) = sw * ps * ps;             // a5  (shared)
      row(7 + 2 * g) = sw * pt;          // d1_g (splay)
      row(8 + 2 * g) = sw * ps;          // d2_g (splay)

      rows.push_back(row);
      rhs.push_back(sw * pn);
      weight_total += w;
      ++count[g];
    }
  }

  if (count[0] < opt.min_per_group || count[1] < opt.min_per_group) {
    return std::nullopt;
  }

  const int ndata = static_cast<int>(rows.size());
  const int ntotal = ndata + detail::kNumPenaltyRows;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ntotal, detail::kNumUnknowns);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(ntotal);
  for (int i = 0; i < ndata; ++i) {
    A.row(i) = rows[i].transpose();
    b(i) = rhs[i];
  }

  // Penalty rows (zero RHS), scaled by the total data weight.
  const double splay_w = std::sqrt(opt.splay_penalty * weight_total);
  const double curv_w = std::sqrt(opt.curvature_penalty * weight_total);
  int prow = ndata;
  for (int c = 7; c < detail::kNumUnknowns; ++c) A(prow++, c) = splay_w;
  for (int c = 4; c < 7; ++c) A(prow++, c) = curv_w;

  const Eigen::VectorXd sol_dyn =
      A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  if (!sol_dyn.allFinite()) return std::nullopt;

  detail::CoeffVector sol;
  for (int c = 0; c < detail::kNumUnknowns; ++c) sol(c) = sol_dyn(c);

  Result out;
  out.count[0] = count[0];
  out.count[1] = count[1];

  // Residual over data rows only, so max_resid measures fit quality.
  double s2 = 0.0;
  for (int i = 0; i < ndata; ++i) {
    const double res = rows[i].dot(sol) - rhs[i];
    s2 += res * res;
  }
  out.rms_residual = std::sqrt(s2 / static_cast<double>(ndata));
  if (out.rms_residual > opt.max_resid) return std::nullopt;

  const double a3 = sol(4);
  const double a4 = sol(5);
  const double a5 = sol(6);
  for (int g = 0; g < 2; ++g) {
    const double ft = sol(2) + sol(7 + 2 * g);
    const double fs = sol(3) + sol(8 + 2 * g);

    IRL::Normal fitted = nref - ft * tref - fs * sref;
    fitted.normalize();
    // Restore each face's sign against its network normal.
    const IRL::Normal& seed = (g == 0) ? seed0 : seed1;
    if (fitted * seed < 0.0) fitted = -fitted;
    out.normal[g] = fitted;

    const double denom = std::pow(1.0 + ft * ft + fs * fs, 1.5);
    out.curvature[g] = -((1.0 + ft * ft) * (2.0 * a5) -
                         2.0 * ft * fs * a4 +
                         (1.0 + fs * fs) * (2.0 * a3)) / denom / opt.mesh_size;
  }

  out.thickness = std::abs(sol(1) - sol(0)) * opt.mesh_size;
  const double opposed = -(out.normal[0] * out.normal[1]);
  out.splay_angle = std::acos(std::max(-1.0, std::min(1.0, opposed)));
  return out;
}

}  // namespace r2pcouple

// ===========================================================================
namespace r2ppass {
// ===========================================================================

struct Options {
  r2psort::Options sort;
  r2pcouple::Options fit;
  double max_splay_change = 0.30;  // rad; reject fits that move the wedge angle more
  bool refine_two_plane = true;

  // Keep the best single plane instead when its film-centroid error is within
  // plane_drop_bias of the two-plane error (cleanReconstruction only drops a
  // plane that misses the cell entirely).
  bool select_plane_count = true;
  double plane_drop_bias = 1.05;
};

struct Stats {
  int visited = 0;
  int refined = 0;
  int split_failed = 0;
  int fit_rejected = 0;
  int splay_rejected = 0;
  int collapsed_to_one = 0;
};

// The 3x3x3 stencil for cell (i,j,k), CENTER FIRST.
inline std::vector<r2pgeom::CellPlanes> buildStencil(
    const BasicMesh& mesh, const Data<double>& a_liquid_volume_fraction,
    const Data<IRL::PlanarSeparator>& snapshot, const int i, const int j,
    const int k) {
  std::vector<r2pgeom::CellPlanes> cells;
  cells.reserve(27);

  for (int pass = 0; pass < 2; ++pass) {
    for (int ii = i - 1; ii <= i + 1; ++ii) {
      for (int jj = j - 1; jj <= j + 1; ++jj) {
        for (int kk = k - 1; kk <= k + 1; ++kk) {
          const bool is_center = (ii == i && jj == j && kk == k);
          // pass 0 takes the center only; pass 1 takes everything else.
          if ((pass == 0) != is_center) continue;

          r2pgeom::CellPlanes cp;
          cp.lo = IRL::Pt(mesh.x(ii), mesh.y(jj), mesh.z(kk));
          cp.hi = IRL::Pt(mesh.x(ii + 1), mesh.y(jj + 1), mesh.z(kk + 1));
          cp.separator = snapshot(ii, jj, kk);
          const double vf = a_liquid_volume_fraction(ii, jj, kk);
          cp.mixed = vf > IRL::global_constants::VF_LOW &&
                     vf < IRL::global_constants::VF_HIGH &&
                     cp.separator.getNumberOfPlanes() > 0;
          cells.push_back(cp);
        }
      }
    }
  }
  return cells;
}

// Distance from a reconstruction's centroid of one phase (0 = liquid, 1 = gas)
// to the target. Every candidate matches the volume fraction, so this is the
// discriminating moment. Callers pass the film phase: for a thin gas film the
// liquid centroid barely moves between candidates.
inline double centroidError(const IRL::RectangularCuboid& cell,
                            const IRL::PlanarSeparator& sep, const int phase,
                            const IRL::Pt& target_centroid) {
  const IRL::SeparatedMoments<IRL::VolumeMoments> moments =
      IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>,
                                      IRL::ReconstructionDefaultCuttingMethod>(
          cell, sep);
  const IRL::Pt c = moments[phase].centroid();
  const double dx = c[0] - target_centroid[0];
  const double dy = c[1] - target_centroid[1];
  const double dz = c[2] - target_centroid[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Sorts the stencil, fits both faces, and writes the refined normals back.
// cells[0] must be the centre. The normals are left unchanged unless this
// returns true.
inline bool refineNormals(const std::vector<r2pgeom::CellPlanes>& cells,
                          IRL::Normal& normal1, IRL::Normal& normal2,
                          const Options& opt, Stats* stats) {
  if (cells.empty() || !cells[0].mixed) return false;
  if (cells[0].separator.getNumberOfPlanes() < 2) return false;

  std::vector<r2pgeom::Tagged> tagged;
  std::vector<std::size_t> cell_begin;
  r2pgeom::flattenStencil(cells, &tagged, &cell_begin);
  if (tagged.size() < 2 * static_cast<std::size_t>(opt.fit.min_per_group)) {
    ++stats->fit_rejected;
    return false;
  }

  const IRL::Normal network0 = normal1;
  const IRL::Normal network1 = normal2;
  r2psort::sortPlanes(tagged, cell_begin, network0, network1, opt.sort);

  double area_fraction[2] = {0.0, 0.0};
  if (!r2psort::groupAreaFractions(tagged, area_fraction)) {
    ++stats->fit_rejected;
    return false;
  }
  if (std::min(area_fraction[0], area_fraction[1]) <
      opt.sort.min_group_area_fraction) {
    ++stats->split_failed;
    return false;
  }

  const std::vector<plicparab::SurfacePolygon> polys0 =
      r2psort::gatherGroup(tagged, 0);
  const std::vector<plicparab::SurfacePolygon> polys1 =
      r2psort::gatherGroup(tagged, 1);
  if (polys0.empty() || polys1.empty()) {
    ++stats->fit_rejected;
    return false;
  }

  const std::optional<r2pcouple::Result> fit =
      r2pcouple::fitCoupled(polys0, polys1, network0, network1, opt.fit);
  if (!fit) {
    ++stats->fit_rejected;
    return false;
  }

  double rotation[2] = {0.0, 0.0};
  IRL::Normal limited[2];
  limited[0] = r2pgeom::limitedRotation(network0, fit->normal[0],
                                        opt.sort.max_rotation, &rotation[0]);
  limited[1] = r2pgeom::limitedRotation(network1, fit->normal[1],
                                        opt.sort.max_rotation, &rotation[1]);

  // Wedge opening before and after (zero for an exact slab).
  const double before = -(network0 * network1);
  const double after = -(limited[0] * limited[1]);
  const double splay_before = std::acos(std::max(-1.0, std::min(1.0, before)));
  const double splay_after = std::acos(std::max(-1.0, std::min(1.0, after)));
  if (std::abs(splay_after - splay_before) > opt.max_splay_change) {
    ++stats->splay_rejected;
    return false;
  }

  normal1 = limited[0];
  normal2 = limited[1];
  return true;
}

// Sweeps the field. a_branch (Data<int> or Data<double>, or nullptr) receives
// a diagnostic tag.
template <class BranchDataType>
inline Stats runImpl(const Data<double>& a_liquid_volume_fraction,
                     const Data<IRL::Pt>& a_liquid_centroid,
                     const Data<IRL::Pt>& a_gas_centroid,
                     Data<IRL::PlanarSeparator>* a_interface,
                     BranchDataType* a_branch, const Options& options) {
  const BasicMesh& mesh = a_liquid_volume_fraction.getMesh();
  Stats stats;
  if (!options.refine_two_plane) return stats;

  // Fit against a snapshot so the result does not depend on sweep order.
  const Data<IRL::PlanarSeparator> snapshot = *a_interface;

  Options opt = options;
  if (opt.fit.mesh_size == 1.0) {
    opt.fit.mesh_size = (mesh.dx() + mesh.dy() + mesh.dz()) / 3.0;
  }

  for (int i = mesh.imin(); i <= mesh.imax(); ++i) {
    for (int j = mesh.jmin(); j <= mesh.jmax(); ++j) {
      for (int k = mesh.kmin(); k <= mesh.kmax(); ++k) {
        const double vf = a_liquid_volume_fraction(i, j, k);
        if (vf <= IRL::global_constants::VF_LOW ||
            vf >= IRL::global_constants::VF_HIGH) {
          continue;
        }

        IRL::PlanarSeparator& sep = (*a_interface)(i, j, k);
        if (sep.getNumberOfPlanes() != 2) continue;

        ++stats.visited;

        IRL::Normal normal1 = sep[0].normal();
        IRL::Normal normal2 = sep[1].normal();
        normal1.normalize();
        normal2.normalize();

        const std::vector<r2pgeom::CellPlanes> cells = buildStencil(
            mesh, a_liquid_volume_fraction, snapshot, i, j, k);

        // On any rejection the pass-1 reconstruction is left untouched. A
        // snapped thin film stays a slab: the fit may rotate it, not open it.
        const bool slab = r2psnap::isSlab(sep);
        if (!refineNormals(cells, normal1, normal2, opt, &stats)) continue;
        if (slab) r2psnap::keepSlab(normal1, normal2);
        ++stats.refined;

        // Rebuild with the fitted normals and pass 1's flip state, then
        // re-solve both distances (the solver sets its own initial guess).
        const double flip_i = sep.isNotFlipped() ? 1.0 : -1.0;
        const IRL::RectangularCuboid cube =
            IRL::RectangularCuboid::fromBoundingPts(
                IRL::Pt(mesh.x(i), mesh.y(j), mesh.z(k)),
                IRL::Pt(mesh.x(i + 1), mesh.y(j + 1), mesh.z(k + 1)));

        sep = IRL::PlanarSeparator::fromTwoPlanes(
            IRL::Plane(normal1, 0.0), IRL::Plane(normal2, 0.0), flip_i);
        one_plane_reason(i, j, k) = 0;
        r2pnewton::R2PNewtonDistanceSolver(vf, a_liquid_centroid(i, j, k),
                                           a_gas_centroid(i, j, k), sep, cube);
        if (sep.getNumberOfPlanes() != 2) one_plane_reason(i, j, k) = 7;
        r2pnopinch::applyAt(mesh, i, j, k, vf, a_liquid_centroid(i, j, k), a_gas_centroid(i, j, k), &sep);
        if (one_plane_reason(i, j, k) == 0 && sep.getNumberOfPlanes() != 2) one_plane_reason(i, j, k) = 5;
        //R2PDistanceSolver(vf, a_liquid_centroid(i, j, k), sep, cube);

        // Is one plane actually better here? Never in a thin film the guard
        // holds for (r2p_edge_topology.h): it continues through the cell; nor
        // in a very thin film's PCA slab (r2p_snap.h pcaSlab).
        if (options.select_plane_count && sep.getNumberOfPlanes() == 2 && film_guard(i, j, k) == 0 &&
            snapped(i, j, k) != 3) {
          // Every candidate is scored on the film phase (gas when flipped).
          const int film = flip_i < 0.0 ? 1 : 0;
          const IRL::Pt& film_centroid =
              film == 1 ? a_gas_centroid(i, j, k) : a_liquid_centroid(i, j, k);
          const double err_two = centroidError(cube, sep, film, film_centroid);

          // Try both fitted normals as the single plane.
          double err_one = -1.0;
          IRL::PlanarSeparator best_one;
          for (int g = 0; g < 2; ++g) {
            IRL::PlanarSeparator cand = IRL::PlanarSeparator::fromOnePlane(
                IRL::Plane((g == 0) ? normal1 : normal2, 0.0));
            IRL::setDistanceToMatchVolumeFraction(cube, vf, &cand);
            const double e = centroidError(cube, cand, film, film_centroid);
            if (err_one < 0.0 || e < err_one) {
              err_one = e;
              best_one = cand;
            }
          }

          if (err_one >= 0.0 && err_one <= options.plane_drop_bias * err_two) {
            sep = best_one;
            one_plane_reason(i, j, k) = 6;
            ++stats.collapsed_to_one;
          }
        }

        if (a_branch != nullptr) (*a_branch)(i, j, k) = 6;
      }
    }
  }

  a_interface->updateBorder();
  return stats;
}

template <class BranchDataType>
inline Stats run(const Data<double>& a_liquid_volume_fraction,
                 const Data<IRL::Pt>& a_liquid_centroid,
                 const Data<IRL::Pt>& a_gas_centroid,
                 Data<IRL::PlanarSeparator>* a_interface,
                 BranchDataType* a_branch,
                 const Options& options = Options()) {
  return runImpl(a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid, a_interface,
                 a_branch, options);
}

inline Stats run(const Data<double>& a_liquid_volume_fraction,
                 const Data<IRL::Pt>& a_liquid_centroid,
                 const Data<IRL::Pt>& a_gas_centroid,
                 Data<IRL::PlanarSeparator>* a_interface,
                 const Options& options = Options()) {
  return runImpl<Data<int>>(a_liquid_volume_fraction, a_liquid_centroid, a_gas_centroid,
                            a_interface, nullptr, options);
}

}  // namespace r2ppass

#endif  // R2P_PARABOLOID_PASS_H_