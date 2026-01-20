// This file is part of the Interface Reconstruction Library (IRL),
// a library for interface reconstruction and computational geometry operations.
//
// Copyright (C) 2025 Valentin Wasquel <wasquel.valentin@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef IRL_CYLINDER_RECONSTRUCTIONS_ALIGNED_PARABOLOID_H_
#define IRL_CYLINDER_RECONSTRUCTIONS_ALIGNED_PARABOLOID_H_

#include <ostream>

namespace IRL {

// Infinit cylinder in the form z^2 + b*y^2 = r (! not r^2 !)
template <class ScalarType>
class AlignedCylinderBase {
 public:
  // using PolynomialBase<2, ScalarType>::PolynomialBase;
  using value_type = ScalarType;

  constexpr AlignedCylinderBase(void) {
    std::fill(coefficients_m.begin(), coefficients_m.end(), ScalarType(0));
    coefficients_m[2] = 1;
  }

  /// @brief Generic constructor for an infinit cylinder alligne with the x axis in the form of z^2 + b*y^2 = r (! not r^2 !)
  /// @param a_coefficients vector containing [b, r] (r is the radius squared !)
  constexpr AlignedCylinderBase(
      const std::array<ScalarType, 2>& a_coefficients) {
    coefficients_m[0] = a_coefficients[0];
    coefficients_m[1] = a_coefficients[1];
    coefficients_m[2] = 1;
  }

  /// @brief Generic constructor for an infinit cylinder alligne with the x axis in the form of z^2 + b*y^2 = r (! not r^2 !)
  /// @param a_coefficients vector containing [b, r, f] (r is the radius squared !, f is the liquid/gas flip (1 or -1))
    constexpr AlignedCylinderBase(
      const std::array<ScalarType, 3>& a_coefficients) {
    coefficients_m = a_coefficients;
  }

  /// @brief Create a new aligned cylinder based on another aligned cylinder
  /// @param a_aligned_cylinder 
  template <class OtherScalarType>
  constexpr AlignedCylinderBase(
      const AlignedCylinderBase<OtherScalarType>& a_aligned_cylinder) {
    coefficients_m[0] = ScalarType(a_aligned_cylinder.b());
    coefficients_m[1] = ScalarType(a_aligned_cylinder.r());
    coefficients_m[2] = ScalarType(a_aligned_cylinder.f());
  };

  ScalarType& b(void) { return coefficients_m[0]; }
  ScalarType b(void) const { return coefficients_m[0]; }
  /// @brief get the radius squared of the infinite cylinder (r = z^2 + b*y^2)
  ScalarType& r(void) { return coefficients_m[1]; }
  ScalarType r(void) const { return coefficients_m[1]; }
  /// @brief get the liquid/gas flip (1 or -1)
  ScalarType& f(void) { return coefficients_m[2]; }
  ScalarType f(void) const { return coefficients_m[2]; }

  void serialize(ByteBuffer* a_buffer) const {
    a_buffer->pack(coefficients_m.data(), 3);
  }
  void unpackSerialized(ByteBuffer* a_buffer) {
    a_buffer->unpack(coefficients_m.data(), 3);
  }

 private:
  std::array<ScalarType, 3> coefficients_m;
};

// Infinit cylinder in the form z^2 + b*y^2 = r (! not r^2 !)
using AlignedCylinder = AlignedCylinderBase<double>;

template <class ScalarType>
inline std::ostream& operator<<(
    std::ostream& out,
    const AlignedCylinderBase<ScalarType>& a_aligned_cylinder) {
  if (a_aligned_cylinder.b() < ScalarType(0)) {
    out << a_aligned_cylinder.r() << " = z^2 " << a_aligned_cylinder.b() << " * y^2" << ", " << a_aligned_cylinder.f() << " flip";
  } else {
    out << a_aligned_cylinder.r() << " = z^2 + " << a_aligned_cylinder.b() << " * y^2" << ", " << a_aligned_cylinder.f() << " flip";
  }
  return out;
}

}  // namespace IRL

#endif  // IRL_CYLINDER_RECONSTRUCTIONS_ALIGNED_PARABOLOID_H_
