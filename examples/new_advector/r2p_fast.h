// Speed-oriented versions of R2P3D_Net and R2P3D_Hybrid.
//
// R2P3D_NetFast computes the same reconstruction as R2P3D_Net with the
// diagnostics, dead code and redundant work removed:
//   - no back-projection of the old interface, colinearity metrics or IRL
//     R2P3D: R2P3D_Net computed all three but only kept R2P3D's answer on the
//     domain-boundary layer. The network is used there too (ghost centroids
//     are shifted by the periodic correction, so its inputs are valid).
//   - the three networks run vectorised, skipping zero inputs, one cell at a
//     time (kBatch in r2p_fast.cpp); outputs are bit-identical to the
//     generated headers' loops.
//   - pass 2 reads the pass-1 field directly and writes its results after the
//     sweep instead of copying the whole field first.
//
// R2P3D_HybridFast computes the same reconstruction as R2P3D_Hybrid without
// the unused colinearity metrics, with the faster networks, and with the
// back-projection restricted to the source cells that can reach an R2P cell.

#ifndef EXAMPLES_NEW_ADVECTOR_R2P_FAST_H_
#define EXAMPLES_NEW_ADVECTOR_R2P_FAST_H_

#include "irl/planar_reconstruction/localized_separator_link.h"
#include "irl/planar_reconstruction/planar_separator.h"

#include "examples/new_advector/data.h"

struct R2P3D_NetFast {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

struct R2P3D_HybridFast {
  static void getReconstruction(
      const Data<double>& a_liquid_volume_fraction,
      const Data<IRL::Pt>& a_liquid_centroid,
      const Data<IRL::Pt>& a_gas_centroid,
      const Data<IRL::LocalizedSeparatorLink>& a_localized_separator_link,
      const double a_dt, const Data<double>& a_U, const Data<double>& a_V,
      const Data<double>& a_W, Data<IRL::PlanarSeparator>* a_interface);
};

namespace r2pfast {

// Per-stage wall time (seconds) and cell counts, accumulated over calls. Only
// filled when compiled with R2PFAST_PROFILE (the timing harness).
struct Profile {
  double stage[12] = {};
  long count[8] = {};
  void reset() { *this = Profile(); }
};
extern Profile g_profile;
extern const char* const kStageNames[12];

}  // namespace r2pfast

#endif  // EXAMPLES_NEW_ADVECTOR_R2P_FAST_H_
