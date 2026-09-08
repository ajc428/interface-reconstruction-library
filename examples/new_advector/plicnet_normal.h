#ifndef PLICNET_NORMAL_H_
#define PLICNET_NORMAL_H_

#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"

#include "examples/new_advector/basic_mesh.h"
#include "examples/new_advector/data.h"
#include "examples/new_advector/plicnet.h"

// PLICNet normal for a single cell.
//
// Extracted verbatim from the plicnet_normal lambda in R2P3D_Net so it can be
// called from the paraboloid pass as well. The lambda captured i/j/k from the
// enclosing loop; this takes them as arguments and is otherwise identical --
// same moment packing, same reflect/permute unwind, same mesh scaling.
//
// Keeping ONE copy of this matters: the dir1/dir2 unwind below has to be the
// exact inverse of whatever plicnet::reflect_moments did, so if that changes,
// a second divergent copy would fail silently with plausible-looking normals.
// If you prefer, replace the lambda in reconstruction_types.cpp with a call
// to this function rather than leaving both in place.
inline IRL::Normal plicnetNormal(const BasicMesh& mesh,
                                 const Data<double>& a_liquid_volume_fraction,
                                 const Data<IRL::Pt>& a_liquid_centroid,
                                 const Data<IRL::Pt>& a_gas_centroid,
                                 const int i, const int j, const int k) {
  double moments_p[189] = {0};
  double m000 = 0, m100 = 0, m010 = 0, m001 = 0;
  double center_p[3] = {0};
  int dir1 = 0, dir2 = 0;
  double n[3] = {0};
  double temp = 0;
  const bool flip_plic = (a_liquid_volume_fraction(i, j, k) >= 0.5);

  for (int ii = i - 1; ii < i + 2; ++ii)
    for (int jj = j - 1; jj < j + 2; ++jj)
      for (int kk = k - 1; kk < k + 2; ++kk) {
        const int idx =
            7 * ((ii + 1 - i) * 9 + (jj + 1 - j) * 3 + (kk + 1 - k));
        if (flip_plic) {
          moments_p[idx] = 1.0 - a_liquid_volume_fraction(ii, jj, kk);
          moments_p[idx + 1] =
              (a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
          moments_p[idx + 2] =
              (a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
          moments_p[idx + 3] =
              (a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
          moments_p[idx + 4] =
              (a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
          moments_p[idx + 5] =
              (a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
          moments_p[idx + 6] =
              (a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
        } else {
          moments_p[idx] = a_liquid_volume_fraction(ii, jj, kk);
          moments_p[idx + 1] =
              (a_liquid_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
          moments_p[idx + 2] =
              (a_liquid_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
          moments_p[idx + 3] =
              (a_liquid_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
          moments_p[idx + 4] =
              (a_gas_centroid(ii, jj, kk)[0] - mesh.xm(ii)) / mesh.dx();
          moments_p[idx + 5] =
              (a_gas_centroid(ii, jj, kk)[1] - mesh.ym(jj)) / mesh.dy();
          moments_p[idx + 6] =
              (a_gas_centroid(ii, jj, kk)[2] - mesh.zm(kk)) / mesh.dz();
        }
        m000 += moments_p[idx];
        m100 += (moments_p[idx + 1] + (ii - i)) * moments_p[idx];
        m010 += (moments_p[idx + 2] + (jj - j)) * moments_p[idx];
        m001 += (moments_p[idx + 3] + (kk - k)) * moments_p[idx];
      }

  // A pure cell in the stencil center would divide by zero here. Callers in
  // the pass already screen on VF_LOW/VF_HIGH, but guard anyway since this is
  // now reachable from more than one place.
  if (m000 <= 0.0) return IRL::Normal(0.0, 0.0, 0.0);

  center_p[0] = m100 / m000;
  center_p[1] = m010 / m000;
  center_p[2] = m001 / m000;

  plicnet::reflect_moments(moments_p, center_p, &dir1, &dir2);
  plicnet::get_normal(moments_p, n);
  IRL::Normal normal = IRL::Normal(n[0], n[1], n[2]);

  switch (dir2) {
    case 1: temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
    case 2: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp; break;
    case 3: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp; break;
    case 4: temp=normal[1]; normal[1]=normal[2]; normal[2]=temp;
            temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
    case 5: temp=normal[0]; normal[0]=normal[2]; normal[2]=temp;
            temp=normal[0]; normal[0]=normal[1]; normal[1]=temp; break;
  }
  switch (dir1) {
    case 1: normal[0]=-normal[0]; break;
    case 2: normal[1]=-normal[1]; break;
    case 3: normal[2]=-normal[2]; break;
    case 4: normal[0]=-normal[0]; normal[1]=-normal[1]; break;
    case 5: normal[0]=-normal[0]; normal[2]=-normal[2]; break;
    case 6: normal[1]=-normal[1]; normal[2]=-normal[2]; break;
    case 7: normal[0]=-normal[0]; normal[1]=-normal[1]; normal[2]=-normal[2]; break;
  }
  if (!flip_plic) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }

  normal[0] *= mesh.dx();
  normal[1] *= mesh.dy();
  normal[2] *= mesh.dz();
  if (normal.calculateMagnitude() > 0.0) normal.normalize();
  return normal;
}

#endif  // PLICNET_NORMAL_H_