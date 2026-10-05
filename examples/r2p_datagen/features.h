// Network inputs and labels, built exactly as R2P3D_Net builds its inputs at
// inference: phase-0 choice (r2pFlip's PCA rule by default), 189 phase-0
// moments, PCA direction of the phase-0 centroids oriented toward their
// centre of mass, canonical reflection with r2pnet::reflect_moments, and the
// same forward transform of the direction. Labels are the face normals
// pointing INTO phase 0, put through that same transform.
//
// Two label formats:
//   legacy    6 values, n1 n2; an absent face is zeros (what the deployed
//             network was trained on; inference tests |n| < 0.85)
//   presence  8 values, n1 n2 p1 p2. Each slot has its own presence flag:
//             1 present, 0 absent, -1 ignore (a sliver face, too small to say
//             whether a second plane helps -- mask it out of the presence
//             loss, keep its normal). Slot = side of the film: a face goes to
//             slot 1 if its normal points along the PCA direction, else slot
//             2, so a face keeps its slot as the other face leaves the cell.
//
// The largest face is always kept, however small: a cut cell has at least one
// plane.

#ifndef EXAMPLES_R2P_DATAGEN_FEATURES_H_
#define EXAMPLES_R2P_DATAGEN_FEATURES_H_

#include <Eigen/Dense>
#include <cmath>
#include <utility>
#include <vector>

#include "irl/parameters/constants.h"

#include "examples/new_advector/r2pnet.h"
#include "examples/r2p_datagen/stencil.h"

namespace r2pgen {

enum class Phase0Rule { kPCA, kFilm, kVFSum };
enum class LabelOrder { kXSign, kPCADot };
enum class LabelFormat { kLegacy, kPresence };

struct FeatureSettings {
  Phase0Rule phase0 = Phase0Rule::kPCA;
  LabelOrder order = LabelOrder::kXSign;   // legacy format only
  LabelFormat format = LabelFormat::kLegacy;
  double min_area = 1.0e-4;       // legacy: smaller face below this counts as absent
  double absent_area = 0.01;      // presence: smaller face below this is absent (p = 0)
  double present_area = 0.05;     // presence: at or above this it is present (p = 1); between: ignore (-1)
};

struct Features {
  double input[192] = {0};
  double label[8] = {0};
  int nlabel = 6;
  bool phase0_gas = false;
  int nfaces = 0;                 // faces written (1 or 2)
  int small_presence = 0;         // presence of the smaller face: 1, 0, or -1 (ignore)
  double area[2] = {0.0, 0.0};    // in label order
  double label_splay = -1.0;      // angle between the two faces (deg; 0 = parallel slab)
};

// Sphericity of the centroid cloud of one phase over the 3^3 moments; -1 with
// fewer than 3 cells holding that phase. Matches phaseSphericity in
// reconstruction_types.cpp.
inline double phaseSphericity(const double* m, bool gas) {
  const double vf_low = IRL::global_constants::VF_LOW;
  std::vector<Vec3> pts;
  for (int c = 0; c < 27; ++c) {
    const double f = gas ? 1.0 - m[7 * c] : m[7 * c];
    if (f <= vf_low) continue;
    const int o = gas ? 4 : 1;
    pts.push_back(cellCentre(3, c) + Vec3(m[7 * c + o], m[7 * c + o + 1], m[7 * c + o + 2]));
  }
  if (pts.size() < 3) return -1.0;
  Vec3 mean = Vec3::Zero();
  for (const auto& p : pts) mean += p;
  mean /= double(pts.size());
  Mat3 cov = Mat3::Zero();
  for (const auto& p : pts) cov += (p - mean) * (p - mean).transpose();
  const Vec3 ev = Eigen::SelfAdjointEigenSolver<Mat3>(cov).eigenvalues();
  return ev(2) > 1.0e-30 ? std::max(0.0, ev(0)) / ev(2) : -1.0;
}

// The forward frame transform R2P3D_Net applies to the PCA direction after
// reflect_moments (sign flips, then axis swaps).
inline void forward(int dir1, int dir2, double* v) {
  switch (dir1) {
    case 1: v[0] = -v[0]; break;
    case 2: v[1] = -v[1]; break;
    case 3: v[2] = -v[2]; break;
    case 4: v[0] = -v[0]; v[1] = -v[1]; break;
    case 5: v[0] = -v[0]; v[2] = -v[2]; break;
    case 6: v[1] = -v[1]; v[2] = -v[2]; break;
    case 7: v[0] = -v[0]; v[1] = -v[1]; v[2] = -v[2]; break;
  }
  switch (dir2) {
    case 1: std::swap(v[0], v[1]); break;
    case 2: std::swap(v[1], v[2]); break;
    case 3: std::swap(v[0], v[2]); break;
    case 4: std::swap(v[0], v[1]); std::swap(v[1], v[2]); break;
    case 5: std::swap(v[0], v[1]); std::swap(v[0], v[2]); break;
  }
}

// Its inverse, exactly as R2P3D_Net undoes the frame on the network output.
inline void inverse(int dir1, int dir2, double* v) {
  switch (dir2) {
    case 1: std::swap(v[0], v[1]); break;
    case 2: std::swap(v[1], v[2]); break;
    case 3: std::swap(v[0], v[2]); break;
    case 4: std::swap(v[1], v[2]); std::swap(v[0], v[1]); break;
    case 5: std::swap(v[0], v[2]); std::swap(v[0], v[1]); break;
  }
  forward(dir1, 0, v);
}

// m3: 7*27 liquid-phase moments of the 3^3 stencil (after noise).
// film_is_gas: for Phase0Rule::kFilm, whether the film phase is the gas (-1: no film).
inline Features buildFeatures(const double* m3, const Faces& faces, int film_is_gas, const FeatureSettings& fs,
                              int* dir1_out = nullptr, int* dir2_out = nullptr) {
  Features F;
  double vfsum = 0.0;
  for (int c = 0; c < 27; ++c) vfsum += m3[7 * c];
  const bool vfsum_rule = vfsum >= 0.5 * 27.0;
  switch (fs.phase0) {
    case Phase0Rule::kVFSum: F.phase0_gas = vfsum_rule; break;
    case Phase0Rule::kFilm: F.phase0_gas = film_is_gas < 0 ? vfsum_rule : film_is_gas == 1; break;
    case Phase0Rule::kPCA: {
      const double sl = phaseSphericity(m3, false), sg = phaseSphericity(m3, true);
      F.phase0_gas = (sl < 0.0 || sg < 0.0) ? vfsum_rule : sg < sl;
      break;
    }
  }

  // Phase-0 moments, PCA direction and its orientation, as in R2P3D_Net.
  double m[189];
  double m000 = 0.0;
  Vec3 m1 = Vec3::Zero();
  std::vector<Vec3> pts;
  for (int c = 0; c < 27; ++c) {
    const double* s = m3 + 7 * c;
    double* d = m + 7 * c;
    d[0] = F.phase0_gas ? 1.0 - s[0] : s[0];
    for (int k = 0; k < 3; ++k) {
      d[1 + k] = F.phase0_gas ? s[4 + k] : s[1 + k];
      d[4 + k] = F.phase0_gas ? s[1 + k] : s[4 + k];
    }
    const Vec3 p = cellCentre(3, c) + Vec3(d[1], d[2], d[3]);
    m000 += d[0];
    m1 += d[0] * p;
    if (d[0] > IRL::global_constants::VF_LOW) pts.push_back(p);
  }
  Vec3 mean = Vec3::Zero();
  for (const auto& p : pts) mean += p;
  if (!pts.empty()) mean /= double(pts.size());
  Mat3 cov = Mat3::Zero();
  for (const auto& p : pts) cov += (p - mean) * (p - mean).transpose();
  Vec3 dir = Eigen::SelfAdjointEigenSolver<Mat3>(cov).eigenvectors().col(0).normalized();
  if (m000 > 0.0 && dir.dot(m1 / m000) < 0.0) dir = -dir;

  double center[3] = {dir.x(), dir.y(), dir.z()};
  int dir1 = 0, dir2 = 0;
  r2pnet::reflect_moments(m, center, &dir1, &dir2);
  forward(dir1, dir2, center);
  for (int k = 0; k < 189; ++k) F.input[k] = m[k];
  for (int k = 0; k < 3; ++k) F.input[189 + k] = center[k];
  if (dir1_out) *dir1_out = dir1;
  if (dir2_out) *dir2_out = dir2;

  // Labels: faces pointing into phase 0, in the canonical frame. The largest
  // face is always kept; the smaller one only above the absent threshold.
  const int big = faces.area[1] > faces.area[0] ? 1 : 0, small = 1 - big;
  const double keep = fs.format == LabelFormat::kLegacy ? fs.min_area : fs.absent_area;
  auto canon = [&](int k) {
    const Vec3 n = F.phase0_gas ? faces.normal[k] : Vec3(-faces.normal[k]);
    double v[3] = {n.x(), n.y(), n.z()};
    forward(dir1, dir2, v);
    return Vec3(v[0], v[1], v[2]);
  };
  Vec3 lab[2] = {canon(big), Vec3::Zero()};
  double area[2] = {faces.area[big], 0.0};
  F.nfaces = 1;
  if (faces.area[small] > keep) {
    lab[1] = canon(small);
    area[1] = faces.area[small];
    F.nfaces = 2;
    F.small_presence = fs.format == LabelFormat::kLegacy || faces.area[small] >= fs.present_area ? 1 : -1;
  }
  const Vec3 pca(center[0], center[1], center[2]);
  double pres[2] = {1.0, double(F.small_presence)};

  bool swap = false;
  if (fs.format == LabelFormat::kPresence) {
    // Slot by side of the film: normal along the PCA direction -> slot 1. If
    // both faces fall on the same side, the more aligned one takes slot 1.
    const double d0 = lab[0].dot(pca), d1 = lab[1].dot(pca);
    if (F.nfaces == 1 || (d0 >= 0.0) != (d1 >= 0.0)) swap = d0 < 0.0;
    else swap = d1 > d0;
  } else if (F.nfaces == 2) {
    if (fs.order == LabelOrder::kXSign) {
      // Legacy data_gen rule: the first normal has a lexicographically
      // positive leading component.
      const double eps = 1.0e-10;
      const Vec3& a = lab[0];
      swap = a.x() < -eps || (std::abs(a.x()) <= eps && (a.y() < -eps || (std::abs(a.y()) <= eps && a.z() < -eps)));
    } else {
      swap = lab[1].dot(pca) > lab[0].dot(pca);
    }
  }
  if (swap) { std::swap(lab[0], lab[1]); std::swap(area[0], area[1]); std::swap(pres[0], pres[1]); }
  if (F.nfaces == 2) F.label_splay = std::acos(std::max(-1.0, std::min(1.0, -lab[0].dot(lab[1])))) * 180.0 / kPi;
  for (int k = 0; k < 2; ++k) {
    for (int d = 0; d < 3; ++d) F.label[3 * k + d] = lab[k][d];
    F.area[k] = area[k];
  }
  if (fs.format == LabelFormat::kPresence) {
    F.nlabel = 8;
    F.label[6] = pres[0];
    F.label[7] = pres[1];
  }
  return F;
}

}  // namespace r2pgen

#endif  // EXAMPLES_R2P_DATAGEN_FEATURES_H_
