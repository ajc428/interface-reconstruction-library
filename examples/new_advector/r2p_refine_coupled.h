#ifndef R2P_REFINE_COUPLED_H_
#define R2P_REFINE_COUPLED_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include "examples/new_advector/r2p_coupled_fit.h"
#include "examples/new_advector/r2p_refine.h"

// Coupled-fit adapter. Everything upstream of the fit -- flattenStencil,
// sortPlanes, groupAreaFractions, gatherGroup -- is reused verbatim from
// r2p_refine.h. Only the fit itself changes: two independent plicparab calls
// become one r2pcouple::fitCoupled call, plus a splay-change guard applied
// on top of the fit's own splay penalty.
namespace r2pfit {

struct CoupledResult {
  bool fitted = false;
  bool split_failed = false;
  bool splay_rejected = false;
  double residual = 0.0;
  double thickness = 0.0;
  double splay_angle = 0.0;
  double curvature[2] = {0.0, 0.0};
  double rotation[2] = {0.0, 0.0};
  double area_fraction[2] = {0.0, 0.0};
  int count[2] = {0, 0};
};

struct CoupledOptions {
  Options gate;             // reuses r2pfit::Options for the sort/gate stage
  r2pcouple::Options fit;   // r2pcouple's own knobs (splay_penalty, etc.)
  double max_splay_change = 0.30;
  // Loose on purpose: with the coupled fit's own splay_penalty doing the real
  // work, this is a backstop, not the primary control. Tighten toward 0.10
  // only if artifacts persist after tuning splay_penalty.
};

// Same contract as refineTwoNormals: cells[0] must be the center cell,
// normal1/normal2 must already be mesh-scaled/normalized/flip-applied on
// entry, and they are left unchanged unless this returns true.
inline bool refineTwoNormalsCoupled(
    const std::vector<CellPlanes>& cells, IRL::Normal& normal1,
    IRL::Normal& normal2, CoupledResult* result = nullptr,
    const CoupledOptions& opt = CoupledOptions()) {
  CoupledResult local;
  CoupledResult& res = result ? *result : local;

  if (cells.empty() || !cells[0].mixed) return false;
  if (cells[0].separator.getNumberOfPlanes() < 2) return false;

  std::vector<detail::Tagged> tagged;
  std::vector<std::size_t> cell_begin;
  flattenStencil(cells, &tagged, &cell_begin);
  if (tagged.size() < 2 * static_cast<std::size_t>(opt.fit.min_per_group)) {
    return false;
  }

  const IRL::Normal network0 = normal1;
  const IRL::Normal network1 = normal2;
  sortPlanes(tagged, cell_begin, network0, network1, opt.gate);

  if (!groupAreaFractions(tagged, res.area_fraction)) return false;
  if (std::min(res.area_fraction[0], res.area_fraction[1]) <
      opt.gate.min_group_area_fraction) {
    res.split_failed = true;
    return false;
  }

  const std::vector<plicparab::SurfacePolygon> polys0 = gatherGroup(tagged, 0);
  const std::vector<plicparab::SurfacePolygon> polys1 = gatherGroup(tagged, 1);
  res.count[0] = static_cast<int>(polys0.size());
  res.count[1] = static_cast<int>(polys1.size());
  // gatherGroup returns empty when that group has no center-cell polygon,
  // which would leave fitCoupled without a reference point for the frame.
  if (polys0.empty() || polys1.empty()) return false;

  const std::optional<r2pcouple::Result> fit =
      r2pcouple::fitCoupled(polys0, polys1, network0, network1, opt.fit);
  if (!fit) return false;

  IRL::Normal limited[2];
  limited[0] = detail::limitedRotation(network0, fit->normal[0],
                                       opt.gate.max_rotation, &res.rotation[0]);
  limited[1] = detail::limitedRotation(network1, fit->normal[1],
                                       opt.gate.max_rotation, &res.rotation[1]);

  // Splay-change backstop, measured after max_rotation clamping since that is
  // the normal actually applied. Both angles are the opening of the wedge,
  // i.e. zero when the two faces are exactly antiparallel (a slab).
  const double before_raw = -(network0 * network1);
  const double after_raw = -(limited[0] * limited[1]);
  const double splay_before =
      std::acos(std::max(-1.0, std::min(1.0, before_raw)));
  const double splay_after =
      std::acos(std::max(-1.0, std::min(1.0, after_raw)));
  if (std::abs(splay_after - splay_before) > opt.max_splay_change) {
    res.splay_rejected = true;
    return false;
  }

  normal1 = limited[0];
  normal2 = limited[1];
  res.fitted = true;
  res.residual = fit->rms_residual;
  res.thickness = fit->thickness;
  res.splay_angle = fit->splay_angle;
  res.curvature[0] = fit->curvature[0];
  res.curvature[1] = fit->curvature[1];
  return true;
}

}  // namespace r2pfit

#endif  // R2P_REFINE_COUPLED_H_