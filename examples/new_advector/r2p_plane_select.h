#ifndef R2P_PLANE_SELECT_H_
#define R2P_PLANE_SELECT_H_

#include <cfloat>
#include <cmath>
#include <optional>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/polyhedrons/rectangular_cuboid.h"
#include "irl/interface_reconstruction_methods/volume_fraction_matching.h"
#include "irl/moments/separated_volume_moments.h"
#include "irl/moments/volume_moments.h"
#include "irl/parameters/constants.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "irl/machine_learning_reconstruction/plic_paraboloid.h"
#include "examples/new_advector/r2p_refine.h"

// Candidate scoring and selection for R2P cells.
//
// Generalizes the build_and_score lambda already used in R2P3D_Net's
// one_plane branch. That scorer is the right arbiter for every decision in
// this pass, because it measures the one thing neither the fit nor the
// distance solver controls: how well a reconstruction reproduces the cell's
// ACTUAL moments. Volume fraction is matched exactly by construction in every
// candidate, so the separated centroids are the only discriminating
// information left -- which makes this the same objective the LM was
// minimizing, evaluated rather than searched.
//
// Applying it to a SET of candidate normals rather than a single one is what
// recovers the LM's two lost behaviours: flat sheets come out flat because a
// candidate that is exactly flat scores exactly zero, and second planes get
// dropped because the one-plane candidate wins outright once it is genuinely
// better.
namespace r2psel {

struct ScoreOptions {
  double vf_tolerance = 1.0e-6;
  // Reject any candidate whose realized VF misses the target. A normal that
  // cannot be distance-matched is not a candidate at all.

  double plane_drop_bias = 1.05;
  // One plane wins if its error is within this factor of the two-plane
  // error. Above 1 favours the simpler model on near-ties, which is what
  // stops marginal second planes from surviving on noise.
};

struct Targets {
  double vf = 0.0;
  IRL::Pt liquid_centroid;
  IRL::Pt gas_centroid;
};

// Which candidate won. Mirrors recon_method / branch tagging so the choice
// can be written to VTK and looked at spatially.
enum class Source {
  kNone = 0,
  kTwoPlane = 1,
  kFitNormal0 = 2,
  kFitNormal1 = 3,
  kPlicNet = 4,
  kMergedParaboloid = 5,
  kExisting = 6,
};

struct Choice {
  IRL::PlanarSeparator separator;
  double error = DBL_MAX;
  Source source = Source::kNone;
  bool valid = false;
};

// Scores ANY separator -- one plane or two -- against the cell's moments.
// Returns DBL_MAX for a candidate that failed to hit the target VF.
inline double scoreSeparator(const IRL::RectangularCuboid& cell,
                             const IRL::PlanarSeparator& sep,
                             const Targets& targets,
                             const ScoreOptions& opt = ScoreOptions()) {
  const IRL::SeparatedMoments<IRL::VolumeMoments> svm =
      IRL::getNormalizedVolumeMoments<IRL::SeparatedMoments<IRL::VolumeMoments>>(
          cell, sep);
  const double cell_volume = cell.calculateVolume();
  const double vf_out = svm[0].volume() / cell_volume;
  if (std::abs(vf_out - targets.vf) > opt.vf_tolerance) return DBL_MAX;

  double err = 0.0;
  if (targets.vf > IRL::global_constants::VF_LOW) {
    err += IRL::magnitude(targets.liquid_centroid - svm[0].centroid());
  }
  if (targets.vf < IRL::global_constants::VF_HIGH) {
    err += IRL::magnitude(targets.gas_centroid - svm[1].centroid());
  }
  return err;
}

// Orients a normal so it points from liquid toward gas, matching the
// convention in the existing one_plane branch. findDistanceOnePlane needs the
// sign to be right, and a candidate that arrives backwards would otherwise be
// scored as garbage rather than as itself.
inline IRL::Normal orientAwayFromLiquid(IRL::Normal normal,
                                        const IRL::Pt& liquid_centroid,
                                        const IRL::Pt& cell_centroid) {
  const IRL::Pt offset(liquid_centroid[0] - cell_centroid[0],
                       liquid_centroid[1] - cell_centroid[1],
                       liquid_centroid[2] - cell_centroid[2]);
  if (normal * offset > 0.0) normal = -normal;
  return normal;
}

// Builds a volume-conserving one-plane separator from a raw normal and scores
// it. Direct transcription of the existing build_and_score lambda.
inline double buildOnePlaneAndScore(const IRL::RectangularCuboid& cell,
                                    IRL::Normal normal, const Targets& targets,
                                    IRL::PlanarSeparator* out,
                                    const ScoreOptions& opt = ScoreOptions()) {
  if (normal.calculateMagnitude() < 0.5) return DBL_MAX;
  normal.normalize();
  const double d = IRL::findDistanceOnePlane(cell, targets.vf, normal);
  *out = IRL::PlanarSeparator::fromOnePlane(IRL::Plane(normal, d));
  return scoreSeparator(cell, *out, targets, opt);
}

// A named candidate normal awaiting scoring.
struct CandidateNormal {
  IRL::Normal normal;
  Source source = Source::kNone;
  bool present = false;
};

// Scores every supplied one-plane candidate and returns the winner.
// Candidates are oriented against the liquid centroid before scoring, so
// callers may pass normals in either sign convention.
inline Choice bestSinglePlane(const IRL::RectangularCuboid& cell,
                              const std::vector<CandidateNormal>& candidates,
                              const Targets& targets,
                              const ScoreOptions& opt = ScoreOptions()) {
  Choice best;
  const IRL::Pt cell_centroid = cell.calculateCentroid();

  for (std::size_t c = 0; c < candidates.size(); ++c) {
    if (!candidates[c].present) continue;
    const IRL::Normal oriented = orientAwayFromLiquid(
        candidates[c].normal, targets.liquid_centroid, cell_centroid);

    IRL::PlanarSeparator sep;
    const double err = buildOnePlaneAndScore(cell, oriented, targets, &sep, opt);
    if (err < best.error) {
      best.error = err;
      best.separator = sep;
      best.source = candidates[c].source;
      best.valid = (err < DBL_MAX);
    }
  }
  return best;
}

// Single paraboloid through the WHOLE stencil, ignoring the two-group sort.
//
// This is the candidate that matters most for a flat sheet that should be one
// plane. The sorted fits each see half the data and disagree slightly; a
// merged fit sees all of it as one surface. fitIntegral's own back-face skip
// discards polygons facing away from the seed, so passing the dominant
// group's normal as seed makes this fit exactly the dominant surface -- which
// is the surface that should survive when the second plane is dropped.
//
// Returns nullopt when there are too few polygons or the fit is rejected.
inline std::optional<IRL::Normal> mergedParaboloidNormal(
    const std::vector<r2pfit::CellPlanes>& cells, const IRL::Normal& seed,
    const double mesh_size, const r2pfit::Options& opt = r2pfit::Options()) {
  std::vector<r2pfit::detail::Tagged> tagged;
  std::vector<std::size_t> cell_begin;
  r2pfit::flattenStencil(cells, &tagged, &cell_begin);
  if (tagged.empty()) return std::nullopt;

  // Center polygon best aligned with the seed goes first: it supplies the
  // reference point and frame for the fit.
  std::vector<plicparab::SurfacePolygon> polys;
  int best_center = -1;
  double best_dot = -2.0;
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    if (!tagged[i].is_center) continue;
    const double d = tagged[i].poly.normal * seed;
    if (d > best_dot) {
      best_dot = d;
      best_center = static_cast<int>(i);
    }
  }
  if (best_center < 0) return std::nullopt;

  polys.push_back(tagged[best_center].poly);
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    if (static_cast<int>(i) == best_center) continue;
    polys.push_back(tagged[i].poly);
  }
  if (static_cast<int>(polys.size()) < opt.parab_minpts) return std::nullopt;

  const std::optional<plicparab::FitResult> fit =
      (opt.orientation_method == 2)
          ? plicparab::fitIntegral(polys, mesh_size, opt.parab_h)
          : plicparab::fitPointwise(polys, mesh_size, opt.parab_h);
  if (!fit) return std::nullopt;
  if (fit->rms_residual > opt.parab_maxresid) return std::nullopt;

  IRL::Normal out = fit->normal;
  if (out * seed < 0.0) out = -out;
  return out;
}

}  // namespace r2psel

#endif  // R2P_PLANE_SELECT_H_