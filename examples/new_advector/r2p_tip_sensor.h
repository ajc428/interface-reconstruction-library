// Film-tip sensor: C++ port of the key idea of NGA2's detect_lig_edge.
//
// Over the 26 neighbours of a cell (the cell itself excluded), take the
// centroids of the film phase, weighted by its volume fraction, and measure
// their spread (weighted second moment, cell^2) about their common centre of
// mass. A film that continues through the cell fills a whole layer of
// neighbours, so its centroids are spread out; at a tip (ligament or sheet
// end) the film reaches only a few neighbours on one side, so the spread is
// small. The sensor fires at spread <= kMaxSpread.
//
// Not ported: the connected-component filter (NGA2 drops neighbours of
// another structure), the droplet exclusion (struct_type 3) and the wall
// checks. The film phase is the one R2P-Net uses (the PCA rule).


#ifndef EXAMPLES_NEW_ADVECTOR_R2P_TIP_SENSOR_H_
#define EXAMPLES_NEW_ADVECTOR_R2P_TIP_SENSOR_H_

#include <cstdlib>

#include "irl/geometry/general/pt.h"
#include "irl/parameters/constants.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"

namespace r2ptip {

// Threshold on the spread (cell^2); R2P_TIP_SPREAD overrides it for a run.
inline double maxSpread() {
  static const double v = [] {
    const char* s = std::getenv("R2P_TIP_SPREAD");
    return (s != nullptr && *s != '\0') ? std::strtod(s, nullptr) : 0.2;
  }();
  return v;
}

// Spread of the film-phase centroids of the 26 neighbours of (i,j,k) about
// their centre of mass, in cell^2 (0 if the neighbours hold no film).
inline double filmSpread(const Data<double>& vf, const Data<IRL::Pt>& liq, const Data<IRL::Pt>& gas,
                         const int i, const int j, const int k, const bool film_is_gas) {
  const BasicMesh& mesh = vf.getMesh();
  double w[26], p[26][3];
  int n = 0;
  double wsum = 0.0, c[3] = {0.0, 0.0, 0.0};
  for (int ii = i - 1; ii <= i + 1; ++ii)
    for (int jj = j - 1; jj <= j + 1; ++jj)
      for (int kk = k - 1; kk <= k + 1; ++kk) {
        if (ii == i && jj == j && kk == k) continue;
        const double f = vf(ii, jj, kk);
        const IRL::Pt& x = film_is_gas ? gas(ii, jj, kk) : liq(ii, jj, kk);
        w[n] = film_is_gas ? 1.0 - f : f;
        // Centroid relative to the centre cell, in cell units
        p[n][0] = (x[0] - mesh.xm(ii)) / mesh.dx() + (ii - i);
        p[n][1] = (x[1] - mesh.ym(jj)) / mesh.dy() + (jj - j);
        p[n][2] = (x[2] - mesh.zm(kk)) / mesh.dz() + (kk - k);
        wsum += w[n];
        for (int d = 0; d < 3; ++d) c[d] += w[n] * p[n][d];
        ++n;
      }
  if (wsum <= IRL::global_constants::VF_LOW) return 0.0;
  for (int d = 0; d < 3; ++d) c[d] /= wsum;
  double spread = 0.0;
  for (int q = 0; q < n; ++q)
    for (int d = 0; d < 3; ++d) spread += w[q] * (p[q][d] - c[d]) * (p[q][d] - c[d]);
  return spread / wsum;
}

// For now the raw spread (cell^2), to pick a threshold: a tip is spread <= maxSpread().
inline double isTip(const Data<double>& vf, const Data<IRL::Pt>& liq, const Data<IRL::Pt>& gas,
                  const int i, const int j, const int k, const bool film_is_gas) {
  return filmSpread(vf, liq, gas, i, j, k, film_is_gas);
}

}  // namespace r2ptip

#endif  // EXAMPLES_NEW_ADVECTOR_R2P_TIP_SENSOR_H_
