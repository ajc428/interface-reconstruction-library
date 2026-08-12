// #ifndef NN_R2P_MOMENT_SOLVER_H_
// #define NN_R2P_MOMENT_SOLVER_H_

// #include <Eigen/Dense>
// #include <algorithm>
// #include <cmath>

// #include "irl/generic_cutting/generic_cutting.h"
// #include "irl/geometry/general/pt.h"
// #include "irl/geometry/general/rotations.h"
// #include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
// #include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
// #include "irl/moments/separated_volume_moments.h"
// #include "irl/moments/volume_moments.h"
// #include "irl/parameters/constants.h"
// #include "irl/planar_reconstruction/planar_separator.h"

// // Refines a two-plane R2P reconstruction produced by a neural network so that
// // it both conserves the cell's target liquid volume fraction exactly and
// // matches the target liquid AND gas centroids as closely as a two-plane
// // "slab" can.
// //
// // The network gives two plane normals but no guarantee that the resulting
// // wedge/slab, once cut against the cell, reproduces the correct centroids.
// // This solver treats the network's normals as an initial guess and searches
// // nearby orientations (small rotations and a wedge-angle change) for the one
// // whose IRL-computed liquid and gas centroids best match the targets,
// // re-deriving the plane distances at every trial so volume is never
// // violated.
// namespace nnr2p {

// // Tuning knobs for the fit. Defaults are conservative.
// struct Options {
//   double regularization = 1.0e-3;
//   // Weight on keeping the fitted normals close to the network's original
//   // normals (i.e. keeping the rotation parameters near zero). Higher values
//   // trust the network more; lower values let the centroid match dominate.

//   double max_rotation = 0.35;
//   // Hard cap, in radians, on both the tilt angle (how far the shared wedge
//   // axis rotates) and the splay angle (how far the wedge half-angle opens or
//   // closes). Keeps the fit from wandering into unphysical orientations.

//   double max_offset = 3.0;
//   // Hard cap on how far the two planes can separate along their bisector,
//   // measured in units of the cell's average side length.

//   bool use_splay = true;
//   // If false, the wedge half-angle between the two planes is frozen at the
//   // network's original value; only the shared tilt and offset are fit.

//   bool allow_plane_drop = true;
//   // If true, run IRL's reconstruction cleaning after the fit so a plane that
//   // ends up outside the cell (or duplicates the other plane) is removed and
//   // the remaining single plane's distance is reset to match the target VF.

//   double volume_tolerance = 1.0e-14;
//   // Volume-fraction tolerance passed to IRL's internal distance solver.

//   double parallel_tolerance = 1.0e-6;
//   // Floor on |normal0 x normal1|^2 below which the two network normals are
//   // treated as parallel/antiparallel; see R2PNormalFit's constructor.
// };

// // Diagnostics returned by solve(), describing how the fit went.
// struct Result {
//   bool converged = false;
//   // True if the optimizer's stopping criteria were met (as opposed to
//   // running out of iterations or stalling).

//   int planes = 2;
//   // Number of planes in the final separator: 2 normally, or 1 if
//   // allow_plane_drop removed one.

//   double centroid_error = 0.0;
//   // (||fitted_liquid_centroid - liquid_bary|| +
//   //  ||fitted_gas_centroid - gas_bary||) / cell_width, i.e. the combined
//   // remaining centroid mismatch, nondimensionalized by the cell size. A
//   // two-plane wedge cannot in general reach arbitrary target centroids
//   // exactly, so this is usually nonzero even at convergence.

//   double volume_error = 0.0;
//   // |achieved_VF - target_VF| for the final separator, after any plane drop.
// };

// namespace detail {

// // Rotates `normal` by `angle` radians about `axis` (right-hand rule) and
// // re-normalizes to remove any drift from the trig evaluation.
// inline IRL::Normal rotate(const IRL::Normal& axis, const double angle,
//                           const IRL::Normal& normal) {
//   IRL::Normal rotated = IRL::UnitQuaternion(angle, axis) * normal;
//   rotated.normalize();
//   return rotated;
// }

// // Returns a unit vector perpendicular to `v`, used to seed a tangent frame.
// // Not IRL::getOrthonormalSystem: that function picks a different
// // perpendicular, which would rotate the tangent frame used below and change
// // the fit specifically when normal0 and normal1 are nearly antiparallel --
// // the common case for R2P sheets and films.
// inline IRL::Normal perpendicularTo(const IRL::Normal& v) {
//   IRL::Normal perpendicular = IRL::crossProduct(
//       v, std::abs(v[0]) < 0.9 ? IRL::Normal(1, 0, 0) : IRL::Normal(0, 1, 0));
//   perpendicular.normalize();
//   return perpendicular;
// }

// }  // namespace detail

// // Owns the cell and the two network-supplied normals, and maps a small
// // parameter vector to a candidate two-plane IRL::PlanarSeparator.
// //
// // Parameter vector layout (size 4 if use_splay, else 3):
// //   params(0), params(1)  -- tilt: rotates both planes together about an
// //                             axis in the tangent plane spanned by
// //                             tilt_axis0_/tilt_axis1_. The pair
// //                             (params(0), params(1)) is itself a rotation
// //                             vector in that 2D tangent basis: its magnitude
// //                             is the tilt angle, its direction picks the
// //                             tilt axis within the tangent plane.
// //   params(2)              -- splay: half-angle adjustment that opens
// //                             (positive) or closes (negative) the wedge
// //                             between the two planes. Omitted if
// //                             use_splay is false.
// //   params(num_params_-1)  -- offset: how far the two planes are pushed
// //                             apart along their shared bisector, in units
// //                             of cell_width_. Only the orientation is
// //                             fit here; IRL fills in the actual plane
// //                             distances needed to hit the target VF.
// //
// // `liquid_bary`/`gas_bary` are the true liquid and gas centroids.
// // IRL::getNormalizedVolumeMoments<SeparatedMoments<VolumeMoments>> always
// // returns the liquid centroid at index 0 and the gas centroid at index 1,
// // regardless of the separator's flip state (flip is already resolved inside
// // IRL's cutting), so no flip-dependent bookkeeping is needed here.
// class R2PNormalFit {
//  public:
//   // `guess` supplies the network's two normals and the cell's flip state.
//   R2PNormalFit(const IRL::RectangularCuboid& cell,
//                const IRL::PlanarSeparator& guess, const double target_fraction,
//                const IRL::Pt& liquid_bary, const IRL::Pt& gas_bary,
//                const Options& options)
//       : cell_(cell),
//         options_(options),
//         target_fraction_(target_fraction),
//         liquid_bary_(liquid_bary),
//         gas_bary_(gas_bary),
//         flip_(guess.isNotFlipped() ? 1.0 : -1.0),
//         cell_width_((cell.calculateSideLength(0) + cell.calculateSideLength(1) +
//                      cell.calculateSideLength(2)) / 3.0),
//         pivot_(cell.calculateCentroid()),
//         num_params_(options.use_splay ? 4 : 3) {
//     normal0_ = guess[0].normal();
//     normal1_ = guess[1].normal();
//     normal0_.normalize();
//     normal1_.normalize();

//     // The bisector of the two normals gives the axis along which the two
//     // planes are pushed apart (the "offset" parameter). If the normals are
//     // exactly opposite this is undefined, so fall back to normal0_.
//     IRL::Normal bisector = normal0_ - normal1_;
//     if (IRL::squaredMagnitude(bisector) < 1.0e-24) bisector = normal0_;
//     bisector.normalize();

//     // tilt_axis0_/tilt_axis1_ span the 2D tangent plane the tilt rotates
//     // within. The natural choice, normal0_ x normal1_, is the axis about
//     // which a single rotation would carry normal0_ onto normal1_ -- but that
//     // axis is itself noise whenever the two normals are nearly parallel or
//     // antiparallel, which is the common R2P case (thin sheets and films).
//     // Below that threshold, fall back to an arbitrary perpendicular of the
//     // bisector instead.
//     tilt_axis0_ = IRL::crossProduct(normal0_, normal1_);
//     if (IRL::squaredMagnitude(tilt_axis0_) < options_.parallel_tolerance)
//       tilt_axis0_ = detail::perpendicularTo(bisector);
//     tilt_axis0_.normalize();
//     tilt_axis1_ = IRL::crossProduct(bisector, tilt_axis0_);
//     tilt_axis1_.normalize();
//   }

//   int numParams() const { return num_params_; }
//   double cellWidth() const { return cell_width_; }
//   const IRL::Pt& pivot() const { return pivot_; }

//   // Computes the two candidate plane normals for a given parameter vector:
//   // first splay the two network normals apart/together about tilt_axis0_,
//   // then rotate the resulting pair together by the tilt.
//   void normals(const Eigen::VectorXd& params, IRL::Normal* out0,
//                IRL::Normal* out1) const {
//     const double tilt_angle = std::hypot(params(0), params(1));
//     IRL::Normal tilt_axis = params(0) * tilt_axis0_ + params(1) * tilt_axis1_;
//     if (tilt_angle > 0.0) tilt_axis.normalize();

//     const double splay = options_.use_splay ? params(2) : 0.0;
//     *out0 = detail::rotate(tilt_axis, tilt_angle,
//                            detail::rotate(tilt_axis0_, splay, normal0_));
//     *out1 = detail::rotate(tilt_axis, tilt_angle,
//                            detail::rotate(tilt_axis0_, -splay, normal1_));
//   }

//   // Builds the full separator for a parameter vector: normals from
//   // normals() above, offset from params(num_params_-1) placing the two
//   // planes symmetrically about the cell centroid (pivot_) along their
//   // bisector, then hands off to IRL to shift both plane distances by a
//   // common amount until the cell's liquid volume fraction matches
//   // target_fraction_ exactly (to within volume_tolerance).
//   IRL::PlanarSeparator build(const Eigen::VectorXd& params) const {
//     IRL::Normal normal0, normal1;
//     normals(params, &normal0, &normal1);
//     const double offset = cell_width_ * params(num_params_ - 1);

//     IRL::PlanarSeparator separator = IRL::PlanarSeparator::fromTwoPlanes(
//         IRL::Plane(normal0, normal0 * pivot_ + offset),
//         IRL::Plane(normal1, normal1 * pivot_ - offset), flip_);
//     IRL::setDistanceToMatchVolumeFraction(cell_, target_fraction_, &separator,
//                                           options_.volume_tolerance);
//     return separator;
//   }

//   // Combined liquid + gas centroid mismatch for an already volume-correct
//   // separator, nondimensionalized by cell_width_.
//   double centroidError(const IRL::PlanarSeparator& separator) const {
//     const auto moments =
//         IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(
//             cell_, separator);
//     const double liquid_error =
//         std::sqrt(IRL::squaredDistanceBetweenPts(moments[0].centroid(), liquid_bary_));
//     const double gas_error =
//         std::sqrt(IRL::squaredDistanceBetweenPts(moments[1].centroid(), gas_bary_));
//     return (liquid_error + gas_error) / cell_width_;
//   }

//   // Residual vector handed to the optimizer: 3 rows for the (nondimensional)
//   // liquid centroid mismatch, 3 more for the gas centroid mismatch, followed
//   // by one row per rotation parameter (everything except the offset)
//   // penalizing departure from the network's original normals. Minimizing
//   // ||residual||^2 balances matching both target centroids against staying
//   // close to what the network predicted.
//   Eigen::VectorXd residual(const Eigen::VectorXd& params) const {
//     const auto moments =
//         IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(
//             cell_, build(params));
//     const IRL::Pt fitted_liquid = moments[0].centroid();
//     const IRL::Pt fitted_gas = moments[1].centroid();
//     const int rotation_params = num_params_ - 1;  // excludes the offset

//     Eigen::VectorXd residual(6 + rotation_params);
//     for (int d = 0; d < 3; ++d) {
//       residual(d)     = (fitted_liquid[d] - liquid_bary_[d]) / cell_width_;
//       residual(3 + d) = (fitted_gas[d] - gas_bary_[d]) / cell_width_;
//     }
//     const double weight = std::sqrt(options_.regularization);
//     for (int j = 0; j < rotation_params; ++j) residual(6 + j) = weight * params(j);
//     return residual;
//   }

//   // Enforces the Options bounds on a parameter vector in place: the tilt
//   // (params 0,1) is clamped as a pair so the tilt angle itself stays under
//   // max_rotation, the splay (params 2, if present) is clamped independently,
//   // and the offset (last entry) is clamped to max_offset.
//   void clamp(Eigen::VectorXd& params) const {
//     const double tilt = std::hypot(params(0), params(1));
//     if (tilt > options_.max_rotation) {
//       params(0) *= options_.max_rotation / tilt;
//       params(1) *= options_.max_rotation / tilt;
//     }
//     if (options_.use_splay)
//       params(2) = std::clamp(params(2), -options_.max_rotation, options_.max_rotation);
//     params(num_params_ - 1) =
//         std::clamp(params(num_params_ - 1), -options_.max_offset, options_.max_offset);
//   }

//  private:
//   IRL::RectangularCuboid cell_;
//   Options options_;
//   double target_fraction_;   // target liquid volume fraction
//   IRL::Pt liquid_bary_;      // target liquid centroid
//   IRL::Pt gas_bary_;         // target gas centroid
//   double flip_;               // +1.0 unflipped, -1.0 flipped
//   double cell_width_;         // average of the cell's three side lengths
//   IRL::Pt pivot_;             // cell centroid; origin for plane distances
//   int num_params_;            // 4 with splay, 3 without
//   IRL::Normal normal0_, normal1_;      // network's original two normals
//   IRL::Normal tilt_axis0_, tilt_axis1_;  // basis for the tilt parameters
// };

// // Minimizes ||model.residual(params)||^2 over `params` in place via
// // Levenberg-Marquardt with Madsen-Nielsen trust-region damping, using
// // forward-difference Jacobians. Returns true if a stopping criterion for
// // convergence was met (small gradient, tiny step, or centroid rows already
// // near zero); false if the damping saturated without a further improving
// // step, i.e. the search stalled. Because a two-plane wedge cannot in general
// // reach arbitrary target centroids exactly, the residual's first six
// // (centroid) rows generally have a nonzero floor even at convergence.
// inline bool runLevenbergMarquardt(const R2PNormalFit& model,
//                                   Eigen::VectorXd& params) {
//   constexpr double kFiniteDifference = 1.0e-6;
//   const int n = model.numParams();

//   model.clamp(params);
//   Eigen::VectorXd residual = model.residual(params);
//   double cost = 0.5 * residual.squaredNorm();
//   double damping = -1.0, growth = 2.0;

//   for (int iteration = 0; iteration < 30; ++iteration) {
//     // Forward-difference Jacobian: column j is d(residual)/d(params(j)).
//     Eigen::MatrixXd jacobian(residual.size(), n);
//     for (int j = 0; j < n; ++j) {
//       Eigen::VectorXd perturbed = params;
//       perturbed(j) += kFiniteDifference;
//       jacobian.col(j) = (model.residual(perturbed) - residual) / kFiniteDifference;
//     }
//     const Eigen::VectorXd gradient = jacobian.transpose() * residual;
//     const Eigen::MatrixXd hessian = jacobian.transpose() * jacobian;
//     const double max_diagonal = hessian.diagonal().maxCoeff();

//     if (gradient.lpNorm<Eigen::Infinity>() < 1.0e-11) return true;
//     if (damping < 0.0) damping = 1.0e-3 * max_diagonal;

//     // Try increasingly large damping (i.e. increasingly conservative,
//     // gradient-descent-like steps) until one actually reduces the cost.
//     bool stepped = false;
//     for (int trial = 0; trial < 12 && !stepped; ++trial) {
//       Eigen::MatrixXd system = hessian;
//       for (int i = 0; i < n; ++i)
//         system(i, i) += damping * std::max(hessian(i, i), 1.0e-10 * max_diagonal);
//       const Eigen::VectorXd step = system.ldlt().solve(-gradient);

//       if (step.norm() < 1.0e-11 * (params.norm() + 1.0e-11)) return true;

//       Eigen::VectorXd trial_params = params + step;
//       model.clamp(trial_params);
//       const Eigen::VectorXd trial_residual = model.residual(trial_params);
//       const double trial_cost = 0.5 * trial_residual.squaredNorm();

//       // gain = actual cost reduction / cost reduction predicted by the
//       // local quadratic model. Near 1 means the step was trustworthy.
//       const double predicted = 0.5 * step.dot(damping * step - gradient);
//       const double gain = predicted > 0.0 ? (cost - trial_cost) / predicted : -1.0;

//       if (gain <= 0.0) {
//         damping *= growth;   // step made things worse; be more conservative
//         growth *= 2.0;
//         continue;
//       }
//       params = trial_params;
//       residual = trial_residual;
//       cost = trial_cost;
//       const double quality = 2.0 * gain - 1.0;
//       damping *= std::max(1.0 / 3.0, 1.0 - quality * quality * quality);
//       growth = 2.0;
//       stepped = true;
//     }
//     if (!stepped) return false;   // damping saturated; no improving step found
//     if (residual.head<6>().norm() < 1.0e-9) return true;  // centroid rows only
//   }
//   return false;
// }

// // Fits a two-plane separator's orientation to best match `liquid_bary` and
// // `gas_bary` simultaneously while conserving `target_fraction` exactly,
// // starting from the network-supplied normals in `interface`. Fitting both
// // centroids (rather than just the liquid one) removes any need to treat
// // flipped and unflipped cells differently: since
// // cell_centroid = VF * liquid_centroid + (1-VF) * gas_centroid always holds,
// // the combined mismatch ||r_liquid||^2 + ||r_gas||^2 depends only on the
// // physical geometry, not on which of the two phases happens to be called
// // "liquid" for a given cell -- so it is inherently symmetric between flip
// // states without any extra scaling.
// inline Result solve(const double target_fraction, const IRL::Pt& liquid_bary,
//                     const IRL::Pt& gas_bary, IRL::PlanarSeparator& interface,
//                     const IRL::RectangularCuboid& cell,
//                     const Options& options = Options()) {
//   Result result;
//   if (target_fraction <= IRL::global_constants::VF_LOW ||
//       target_fraction >= IRL::global_constants::VF_HIGH) {
//     // Cell is (numerically) pure liquid or pure gas; nothing to fit.
//     result.converged = true;
//     return result;
//   }

//   // The region strictly between the two planes is the liquid when unflipped
//   // and the gas when flipped (see IRL::PlanarSeparator's flip convention).
//   // The offset parameter's initial guess is seeded from whichever centroid
//   // that is, since it's the offset that positions the wedge relative to the
//   // blob it encloses.
//   const bool flipped = interface.isFlipped();
//   const IRL::Pt& enclosed_bary = flipped ? gas_bary : liquid_bary;

//   R2PNormalFit model(cell, interface, target_fraction, liquid_bary, gas_bary,
//                      options);
//   Eigen::VectorXd params = Eigen::VectorXd::Zero(model.numParams());
//   {
//     // Seed the offset parameter (the last entry of params) so the two
//     // planes start out roughly straddling enclosed_bary along their shared
//     // axis, rather than starting coincident at the cell centroid.
//     IRL::Normal normal0, normal1;
//     model.normals(params, &normal0, &normal1);
//     IRL::Normal axis = normal0 - normal1;
//     if (IRL::squaredMagnitude(axis) < 1.0e-24) axis = normal0;
//     axis.normalize();
//     params(model.numParams() - 1) =
//         axis * IRL::Pt(enclosed_bary - model.pivot()) / model.cellWidth();
//     model.clamp(params);
//   }

//   result.converged = runLevenbergMarquardt(model, params);

//   IRL::PlanarSeparator separator = model.build(params);
//   if (options.allow_plane_drop) {
//     // Removes any plane that ends up outside the cell (or duplicates the
//     // other plane), then resets the remaining plane's distance so it alone
//     // reproduces target_fraction.
//     IRL::cleanReconstruction(cell, target_fraction, &separator);
//   }

//   interface = separator;
//   result.planes = static_cast<int>(separator.getNumberOfPlanes());
//   result.centroid_error = model.centroidError(separator);
//   result.volume_error = std::abs(
//       IRL::getVolumeFraction<IRL::ReconstructionDefaultCuttingMethod>(
//           cell, interface) - target_fraction);
//   return result;
// }

// }  // namespace nnr2p

// // Entry point used by the reconstruction driver. `VF_target` is the target
// // liquid volume fraction, `liquid_bary_target`/`gas_bary_target` the target
// // liquid and gas centroids, and `a_interface` on input holds the network's
// // initial two-plane guess (whose flip state and normals are used as the
// // starting point) and on output holds the fitted separator.
// inline void R2PDistanceSolver2(double VF_target, IRL::Pt liquid_bary_target,
//                                IRL::Pt gas_bary_target,
//                                IRL::PlanarSeparator& a_interface,
//                                IRL::RectangularCuboid cell) {
//   nnr2p::solve(VF_target, liquid_bary_target, gas_bary_target, a_interface, cell);
// }

// #endif  // NN_R2P_MOMENT_SOLVER_H_







#ifndef NN_R2P_MOMENT_SOLVER_H_
#define NN_R2P_MOMENT_SOLVER_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/general/rotations.h"
#include "irl/interface_reconstruction_methods/r2p_neighborhood.h"
#include "irl/interface_reconstruction_methods/reconstruction_cleaning.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/moments/separated_volume_moments.h"
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

// Refines a two-plane R2P reconstruction produced by a neural network against
// an entire 3x3x3 stencil of cells, not just the center cell, so that it
// conserves the center cell's target liquid volume fraction exactly while
// matching the liquid AND gas centroids of every cell in the stencil as
// closely as a two-plane "slab" can.
//
// The network gives two plane normals but no guarantee that the resulting
// wedge/slab, once cut against the cell, reproduces the correct centroids --
// and a plane orientation fit only to the center cell is often ambiguous
// (many orientations conserve the same volume and centroid there). Cutting
// the SAME pair of planes against every neighbor in the stencil and
// comparing each neighbor's resulting moments to its own known target moments
// gives the fit far more signal to pin down the orientation, exactly as
// classic neighborhood-based PLIC/R2P methods do.
namespace nnr2p {

// Tuning knobs for the fit. Defaults are conservative.
struct Options {
  double regularization = 1.0e-2;
  // Weight on keeping the fitted normals close to the network's original
  // normals (i.e. keeping the rotation parameters near zero). Higher values
  // trust the network more; lower values let the centroid/volume match
  // across the stencil dominate. Internally rescaled by the ratio of this
  // fit's data-row count to a 6-row reference (see R2PNormalFit), so this
  // value means roughly the same thing -- "how much to trust the network
  // relative to the data" -- whether the stencil is 3x3x3, larger, or a
  // single cell; it should not need re-tuning if the stencil size changes.

  double max_rotation = 0.35;
  // Hard cap, in radians, on both the tilt angle (how far the shared wedge
  // axis rotates) and the splay angle (how far the wedge half-angle opens or
  // closes). Keeps the fit from wandering into unphysical orientations.

  double max_offset = 3.0;
  // Hard cap on how far the two planes can separate along their bisector,
  // measured in units of the center cell's average side length.

  bool use_splay = true;
  // If false, the wedge half-angle between the two planes is frozen at the
  // network's original value; only the shared tilt and offset are fit.

  bool allow_plane_drop = true;
  // If true, run IRL's reconstruction cleaning after the fit so a plane that
  // ends up outside the center cell (or duplicates the other plane) is
  // removed and the remaining single plane's distance is reset to match the
  // center cell's target VF.

  double volume_tolerance = 1.0e-14;
  // Volume-fraction tolerance passed to IRL's internal distance solver.

  double parallel_tolerance = 1.0e-6;
  // Floor on |normal0 x normal1|^2 below which the two network normals are
  // treated as parallel/antiparallel; see R2PNormalFit's constructor.
};

// Diagnostics returned by solve(), describing how the fit went.
struct Result {
  bool converged = false;
  // True if the optimizer's stopping criteria were met (as opposed to
  // running out of iterations or stalling).

  int planes = 2;
  // Number of planes in the final separator: 2 normally, or 1 if
  // allow_plane_drop removed one.

  double centroid_error = 0.0;
  // Combined liquid + gas centroid mismatch summed over every cell in the
  // stencil, nondimensionalized by the center cell's width. A two-plane
  // wedge cannot in general reach every neighbor's target exactly, so this
  // is usually nonzero even at convergence.

  double volume_error = 0.0;
  // |achieved_VF - target_VF| for the CENTER cell of the final separator,
  // after any plane drop. This is the one quantity solve() enforces exactly
  // by construction; every other cell's volume is only fit approximately as
  // part of the residual.
};

namespace detail {

// Rotates `normal` by `angle` radians about `axis` (right-hand rule) and
// re-normalizes to remove any drift from the trig evaluation.
inline IRL::Normal rotate(const IRL::Normal& axis, const double angle,
                          const IRL::Normal& normal) {
  IRL::Normal rotated = IRL::UnitQuaternion(angle, axis) * normal;
  rotated.normalize();
  return rotated;
}

// Returns a unit vector perpendicular to `v`, used to seed a tangent frame.
// Not IRL::getOrthonormalSystem: that function picks a different
// perpendicular, which would rotate the tangent frame used below and change
// the fit specifically when normal0 and normal1 are nearly antiparallel --
// the common case for R2P sheets and films.
inline IRL::Normal perpendicularTo(const IRL::Normal& v) {
  IRL::Normal perpendicular = IRL::crossProduct(
      v, std::abs(v[0]) < 0.9 ? IRL::Normal(1, 0, 0) : IRL::Normal(0, 1, 0));
  perpendicular.normalize();
  return perpendicular;
}

}  // namespace detail

// Owns the stencil and the two network-supplied normals, and maps a small
// parameter vector to a candidate two-plane IRL::PlanarSeparator.
//
// Parameter vector layout (size 4 if use_splay, else 3):
//   params(0), params(1)  -- tilt: rotates both planes together about an
//                             axis in the tangent plane spanned by
//                             tilt_axis0_/tilt_axis1_. The pair
//                             (params(0), params(1)) is itself a rotation
//                             vector in that 2D tangent basis: its magnitude
//                             is the tilt angle, its direction picks the
//                             tilt axis within the tangent plane.
//   params(2)              -- splay: half-angle adjustment that opens
//                             (positive) or closes (negative) the wedge
//                             between the two planes. Omitted if
//                             use_splay is false.
//   params(num_params_-1)  -- offset: how far the two planes are pushed
//                             apart along their shared bisector, in units
//                             of cell_width_. Only the orientation is
//                             fit here; IRL fills in the actual plane
//                             distances needed to hit the center cell's
//                             target VF.
//
// The planes are always placed relative to the CENTER cell's centroid
// (pivot_) and sized in units of the CENTER cell's width, then evaluated by
// cutting every cell in the stencil with that same pair of planes -- so the
// fit finds one orientation that is simultaneously plausible for the whole
// neighborhood, not just the center cell.
//
// IRL::getNormalizedVolumeMoments<SeparatedMoments<VolumeMoments>> always
// returns the liquid centroid at index 0 and the gas centroid at index 1,
// regardless of a separator's flip state (flip is already resolved inside
// IRL's cutting), so no flip-dependent bookkeeping is needed here.
class R2PNormalFit {
 public:
  // `neighborhood` supplies both the cells to cut and, per cell, the target
  // SeparatedMoments<VolumeMoments> to match; it must outlive this object.
  // `guess` supplies the network's two normals and the flip state to use for
  // every cut in the stencil. `enclosed_bary` is the centroid of whichever
  // phase lies between the two planes (see solve()); it fixes both the
  // initial and the regularization target for the offset parameter.
  R2PNormalFit(const IRL::R2PNeighborhood<IRL::RectangularCuboid>& neighborhood,
               const IRL::PlanarSeparator& guess, const IRL::Pt& enclosed_bary,
               const Options& options)
      : neighborhood_(neighborhood),
        options_(options),
        flip_(guess.isNotFlipped() ? 1.0 : -1.0),
        cell_width_((neighborhood.getCenterCell().calculateSideLength(0) +
                     neighborhood.getCenterCell().calculateSideLength(1) +
                     neighborhood.getCenterCell().calculateSideLength(2)) /
                    3.0),
        pivot_(neighborhood.getCenterCell().calculateCentroid()),
        target_fraction_(neighborhood.getCenterCellStoredMoments()[0].volume() /
                         neighborhood.getCenterCell().calculateVolume()),
        num_params_(options.use_splay ? 4 : 3) {
    normal0_ = guess[0].normal();
    normal1_ = guess[1].normal();
    normal0_.normalize();
    normal1_.normalize();

    // The bisector of the two normals gives the axis along which the two
    // planes are pushed apart (the "offset" parameter). If the normals are
    // exactly opposite this is undefined, so fall back to normal0_.
    IRL::Normal bisector = normal0_ - normal1_;
    if (IRL::squaredMagnitude(bisector) < 1.0e-24) bisector = normal0_;
    bisector.normalize();

    // tilt_axis0_/tilt_axis1_ span the 2D tangent plane the tilt rotates
    // within. The natural choice, normal0_ x normal1_, is the axis about
    // which a single rotation would carry normal0_ onto normal1_ -- but that
    // axis is itself noise whenever the two normals are nearly parallel or
    // antiparallel, which is the common R2P case (thin sheets and films).
    // Below that threshold, fall back to an arbitrary perpendicular of the
    // bisector instead.
    tilt_axis0_ = IRL::crossProduct(normal0_, normal1_);
    if (IRL::squaredMagnitude(tilt_axis0_) < options_.parallel_tolerance)
      tilt_axis0_ = detail::perpendicularTo(bisector);
    tilt_axis0_.normalize();
    tilt_axis1_ = IRL::crossProduct(bisector, tilt_axis0_);
    tilt_axis1_.normalize();

    // Where the offset parameter starts: the same bisector direction used to
    // place the two planes, projected onto how far enclosed_bary sits from
    // the center cell's centroid. Seeding only -- NOT regularized toward,
    // since it is a translation-like quantity, not a separation/width
    // quantity comparable to what the offset parameter controls.
    offset_seed_ = bisector * IRL::Pt(enclosed_bary - pivot_) / cell_width_;

    // regularization_weight_ scales with sqrt(data rows), so options_.
    // regularization keeps the same relative meaning regardless of stencil
    // size. Reference is 6 rows -- 3 liquid + 3 gas centroid, no volume row,
    // no neighbors -- the single-cell centroid-only fit this scale was
    // originally tuned against. Without this, a value tuned for one stencil
    // size stops being meaningful at another: a 3x3x3 stencil has 7*27=189
    // data rows against this fit's 3-4 regularization rows, versus 6 data
    // rows in the single-cell case, so the *same* nominal regularization
    // value exerts roughly 31.5x less relative pull toward the network's
    // guess unless compensated for here.
    const double data_rows = 7.0 * static_cast<double>(neighborhood.size());
    regularization_weight_ = std::sqrt(options_.regularization * data_rows);
  }

  int numParams() const { return num_params_; }
  double cellWidth() const { return cell_width_; }
  const IRL::Pt& pivot() const { return pivot_; }
  double targetFraction() const { return target_fraction_; }
  // offset_seed_ is retained only as the initial guess for that parameter
  // (see solve()); it does not appear in residual().
  double offsetSeed() const { return offset_seed_; }

  // Computes the two candidate plane normals for a given parameter vector:
  // first splay the two network normals apart/together about tilt_axis0_,
  // then rotate the resulting pair together by the tilt.
  void normals(const Eigen::VectorXd& params, IRL::Normal* out0,
               IRL::Normal* out1) const {
    const double tilt_angle = std::hypot(params(0), params(1));
    IRL::Normal tilt_axis = params(0) * tilt_axis0_ + params(1) * tilt_axis1_;
    if (tilt_angle > 0.0) tilt_axis.normalize();

    const double splay = options_.use_splay ? params(2) : 0.0;
    *out0 = detail::rotate(tilt_axis, tilt_angle,
                           detail::rotate(tilt_axis0_, splay, normal0_));
    *out1 = detail::rotate(tilt_axis, tilt_angle,
                           detail::rotate(tilt_axis0_, -splay, normal1_));
  }

  // Builds the full separator for a parameter vector: normals from
  // normals() above, offset from params(num_params_-1) placing the two
  // planes symmetrically about the center cell's centroid (pivot_) along
  // their bisector, then hands off to IRL to shift both plane distances by a
  // common amount until the CENTER cell's liquid volume fraction matches
  // target_fraction_ exactly (to within volume_tolerance). This is the only
  // place volume is enforced exactly; every other cell in the stencil only
  // enters through the residual below.
  IRL::PlanarSeparator build(const Eigen::VectorXd& params) const {
    IRL::Normal normal0, normal1;
    normals(params, &normal0, &normal1);
    const double offset = cell_width_ * params(num_params_ - 1);

    IRL::PlanarSeparator separator = IRL::PlanarSeparator::fromTwoPlanes(
        IRL::Plane(normal0, normal0 * pivot_ + offset),
        IRL::Plane(normal1, normal1 * pivot_ - offset), flip_);
    IRL::setDistanceToMatchVolumeFraction(neighborhood_.getCenterCell(),
                                          target_fraction_, &separator,
                                          options_.volume_tolerance);
    return separator;
  }

  // Combined liquid + gas centroid mismatch, summed over every cell in the
  // stencil, for an already-built separator. Nondimensionalized by
  // cell_width_. A phase with (near-)zero target volume has no physically
  // meaningful centroid -- see residual() -- so its contribution is skipped
  // rather than compared against whatever value happens to be stored there.
  double centroidError(const IRL::PlanarSeparator& separator) const {
    double total = 0.0;
    for (const auto& member : neighborhood_) {
      const auto fitted = member.calculateNormalizedVolumeMoments(separator);
      const auto& target = member.getStoredMoments();
      const double purity_floor =
          IRL::global_constants::VF_LOW * member.getCell().calculateVolume();
      if (target[0].volume() > purity_floor)
        total += std::sqrt(
            IRL::squaredDistanceBetweenPts(fitted[0].centroid(), target[0].centroid()));
      if (target[1].volume() > purity_floor)
        total += std::sqrt(
            IRL::squaredDistanceBetweenPts(fitted[1].centroid(), target[1].centroid()));
    }
    return total / cell_width_;
  }

  // Residual vector handed to the optimizer: for every cell in the stencil,
  // 1 row for the (dimensionless) liquid volume fraction mismatch and 3+3
  // rows for the (nondimensional) liquid/gas centroid mismatch, followed by
  // one row per rotation parameter (everything except the offset) penalizing
  // departure from the network's original normals. The offset is NOT
  // regularized: offset_seed_ is a translation-like quantity (how far the
  // target centroid sits from the cell center), not a separation/width
  // quantity comparable to what the offset parameter actually controls
  // (how far apart the two planes are pushed), so pulling the fit back
  // toward it distorts the wedge width rather than usefully constraining
  // it -- confirmed by regression when this was tried.
  //
  // A phase with (near-)zero target volume has its 3 centroid rows set to
  // zero rather than compared: many VOF solvers leave a pure-phase cell's
  // absent-phase centroid at a sentinel value (often the cell's own
  // centroid) rather than (0,0,0), and comparing that directly against the
  // fitted centroid -- which IRL correctly reports as (0,0,0) for an empty
  // cut -- injects a spurious, mesh-scale mismatch that dominates the real,
  // small signal from the stencil's genuinely mixed cells. Minimizing
  // ||residual||^2 balances matching the whole stencil's known moments
  // against staying close to what the network predicted.
  Eigen::VectorXd residual(const Eigen::VectorXd& params) const {
    const IRL::PlanarSeparator separator = build(params);
    const int rotation_params = num_params_ - 1;  // excludes the offset
    const int stencil_size = static_cast<int>(neighborhood_.size());

    Eigen::VectorXd residual(7 * stencil_size + rotation_params);
    int row = 0;
    for (const auto& member : neighborhood_) {
      const auto fitted = member.calculateNormalizedVolumeMoments(separator);
      const auto& target = member.getStoredMoments();
      const double cell_volume = member.getCell().calculateVolume();
      const double purity_floor = IRL::global_constants::VF_LOW * cell_volume;
      const bool liquid_meaningful = target[0].volume() > purity_floor;
      const bool gas_meaningful = target[1].volume() > purity_floor;

      residual(row++) = (fitted[0].volume() - target[0].volume()) / cell_volume;
      for (int d = 0; d < 3; ++d)
        residual(row++) = liquid_meaningful
            ? (fitted[0].centroid()[d] - target[0].centroid()[d]) / cell_width_
            : 0.0;
      for (int d = 0; d < 3; ++d)
        residual(row++) = gas_meaningful
            ? (fitted[1].centroid()[d] - target[1].centroid()[d]) / cell_width_
            : 0.0;
    }

    for (int j = 0; j < rotation_params; ++j) residual(row++) = regularization_weight_ * params(j);
    return residual;
  }

  // Enforces the Options bounds on a parameter vector in place: the tilt
  // (params 0,1) is clamped as a pair so the tilt angle itself stays under
  // max_rotation, the splay (params 2, if present) is clamped independently,
  // and the offset (last entry) is clamped to max_offset.
  void clamp(Eigen::VectorXd& params) const {
    const double tilt = std::hypot(params(0), params(1));
    if (tilt > options_.max_rotation) {
      params(0) *= options_.max_rotation / tilt;
      params(1) *= options_.max_rotation / tilt;
    }
    if (options_.use_splay)
      params(2) = std::clamp(params(2), -options_.max_rotation, options_.max_rotation);
    params(num_params_ - 1) =
        std::clamp(params(num_params_ - 1), -options_.max_offset, options_.max_offset);
  }

 private:
  const IRL::R2PNeighborhood<IRL::RectangularCuboid>& neighborhood_;
  Options options_;
  double flip_;                // +1.0 unflipped, -1.0 flipped
  double cell_width_;          // average of the center cell's three side lengths
  IRL::Pt pivot_;               // center cell centroid; origin for plane distances
  double target_fraction_;      // center cell's target liquid volume fraction
  double offset_seed_;          // offset's initial value and regularization target
  double regularization_weight_;  // sqrt-scaled by data-row count; see constructor
  int num_params_;              // 4 with splay, 3 without
  IRL::Normal normal0_, normal1_;      // network's original two normals
  IRL::Normal tilt_axis0_, tilt_axis1_;  // basis for the tilt parameters
};

// Minimizes ||model.residual(params)||^2 over `params` in place via
// Levenberg-Marquardt with Madsen-Nielsen trust-region damping, using
// forward-difference Jacobians. Returns true if a stopping criterion for
// convergence was met (small gradient, tiny step, or centroid/volume rows
// already near zero); false if the damping saturated without a further
// improving step, i.e. the search stalled. Because a single two-plane wedge
// cannot in general reach every cell's target exactly, the residual's
// centroid/volume rows generally have a nonzero floor even at convergence.
inline bool runLevenbergMarquardt(const R2PNormalFit& model,
                                  Eigen::VectorXd& params) {
  constexpr double kFiniteDifference = 1.0e-6;
  const int n = model.numParams();

  model.clamp(params);
  Eigen::VectorXd residual = model.residual(params);
  const int fit_rows = residual.size() - (n - 1);  // everything but regularization
  double cost = 0.5 * residual.squaredNorm();
  double damping = -1.0, growth = 2.0;

  for (int iteration = 0; iteration < 30; ++iteration) {
    // Forward-difference Jacobian: column j is d(residual)/d(params(j)).
    Eigen::MatrixXd jacobian(residual.size(), n);
    for (int j = 0; j < n; ++j) {
      Eigen::VectorXd perturbed = params;
      perturbed(j) += kFiniteDifference;
      jacobian.col(j) = (model.residual(perturbed) - residual) / kFiniteDifference;
    }
    const Eigen::VectorXd gradient = jacobian.transpose() * residual;
    const Eigen::MatrixXd hessian = jacobian.transpose() * jacobian;
    const double max_diagonal = hessian.diagonal().maxCoeff();

    if (gradient.lpNorm<Eigen::Infinity>() < 1.0e-11) return true;
    if (damping < 0.0) damping = 1.0e-3 * max_diagonal;

    // Try increasingly large damping (i.e. increasingly conservative,
    // gradient-descent-like steps) until one actually reduces the cost.
    bool stepped = false;
    for (int trial = 0; trial < 12 && !stepped; ++trial) {
      Eigen::MatrixXd system = hessian;
      for (int i = 0; i < n; ++i)
        system(i, i) += damping * std::max(hessian(i, i), 1.0e-10 * max_diagonal);
      const Eigen::VectorXd step = system.ldlt().solve(-gradient);

      if (step.norm() < 1.0e-11 * (params.norm() + 1.0e-11)) return true;

      Eigen::VectorXd trial_params = params + step;
      model.clamp(trial_params);
      const Eigen::VectorXd trial_residual = model.residual(trial_params);
      const double trial_cost = 0.5 * trial_residual.squaredNorm();

      // gain = actual cost reduction / cost reduction predicted by the
      // local quadratic model. Near 1 means the step was trustworthy.
      const double predicted = 0.5 * step.dot(damping * step - gradient);
      const double gain = predicted > 0.0 ? (cost - trial_cost) / predicted : -1.0;

      if (gain <= 0.0) {
        damping *= growth;   // step made things worse; be more conservative
        growth *= 2.0;
        continue;
      }
      params = trial_params;
      residual = trial_residual;
      cost = trial_cost;
      const double quality = 2.0 * gain - 1.0;
      damping *= std::max(1.0 / 3.0, 1.0 - quality * quality * quality);
      growth = 2.0;
      stepped = true;
    }
    if (!stepped) return false;   // damping saturated; no improving step found
    if (residual.head(fit_rows).norm() < 1.0e-9) return true;
  }
  return false;
}

// Fits a two-plane separator's orientation to best match every cell in
// `neighborhood` while conserving the center cell's volume fraction exactly,
// starting from the network-supplied normals in `interface`. `interface`'s
// flip state is used as the flip state for every cut in the stencil.
inline Result solve(const IRL::R2PNeighborhood<IRL::RectangularCuboid>& neighborhood,
                    IRL::PlanarSeparator& interface,
                    const Options& options = Options()) {
  Result result;
  const auto& center_cell = neighborhood.getCenterCell();
  const auto& center_target = neighborhood.getCenterCellStoredMoments();
  const double target_fraction =
      center_target[0].volume() / center_cell.calculateVolume();

  if (target_fraction <= IRL::global_constants::VF_LOW ||
      target_fraction >= IRL::global_constants::VF_HIGH) {
    // Cell is (numerically) pure liquid or pure gas; nothing to fit.
    result.converged = true;
    return result;
  }

  // The region strictly between the two planes is the liquid when unflipped
  // and the gas when flipped (see IRL::PlanarSeparator's flip convention).
  // The offset parameter's initial guess is seeded from whichever centroid
  // that is, since it's the offset that positions the wedge relative to the
  // blob it encloses.
  const bool flipped = interface.isFlipped();
  const IRL::Pt& enclosed_bary =
      flipped ? center_target[1].centroid() : center_target[0].centroid();

  R2PNormalFit model(neighborhood, interface, enclosed_bary, options);
  Eigen::VectorXd params = Eigen::VectorXd::Zero(model.numParams());
  // Seed the offset at the same value it's regularized toward, so the two
  // planes start out roughly straddling enclosed_bary along their shared
  // bisector rather than starting coincident at the cell centroid.
  params(model.numParams() - 1) = model.offsetSeed();
  model.clamp(params);

  result.converged = runLevenbergMarquardt(model, params);

  IRL::PlanarSeparator separator = model.build(params);
  if (options.allow_plane_drop) {
    // Removes any plane that ends up outside the center cell (or duplicates
    // the other plane), then resets the remaining plane's distance so it
    // alone reproduces the center cell's target_fraction.
    IRL::cleanReconstruction(center_cell, target_fraction, &separator);
  }

  interface = separator;
  result.planes = static_cast<int>(separator.getNumberOfPlanes());
  result.centroid_error = model.centroidError(separator);
  result.volume_error = std::abs(
      IRL::getVolumeFraction<IRL::ReconstructionDefaultCuttingMethod>(
          center_cell, interface) - target_fraction);
  return result;
}

}  // namespace nnr2p

// Entry point used by the reconstruction driver. `neighborhood` must already
// hold the 3x3x3 stencil's cells and target SeparatedMoments<VolumeMoments>,
// with the center cell registered via setCenterOfStencil(). `a_interface` on
// input holds the network's initial two-plane guess (whose flip state and
// normals are used as the starting point for every cut) and on output holds
// the fitted separator.
inline void R2PDistanceSolver2(
    const IRL::R2PNeighborhood<IRL::RectangularCuboid>& neighborhood,
    IRL::PlanarSeparator& a_interface) {
  nnr2p::solve(neighborhood, a_interface);
}

#endif  // NN_R2P_MOMENT_SOLVER_H_