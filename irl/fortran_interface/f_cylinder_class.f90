!  This file is part of the Interface Reconstruction Library (IRL),
!  a library for interface reconstruction and computational geometry operations.
!
!  Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
!
!  This Source Code Form is subject to the terms of the Mozilla Public
!  License, v. 2.0. If a copy of the MPL was not distributed with this
!  file, You can obtain one at https://mozilla.org/MPL/2.0/.

module f_Cylinder_class
    use, intrinsic :: iso_c_binding
    use f_DefinedTypes
    use f_Poly_class
    use f_PlanarSep_class
    use f_RectCub_class
    use f_SeparatorVariant_class
    implicit none
  
    type, public, bind(C) :: c_Cylinder
      type(C_PTR), private :: object = C_NULL_PTR
      logical(C_BOOL), private :: is_owning  = .false.
    end type c_Cylinder
  
    type, public :: Cylinder_type
      type(c_Cylinder) :: c_object
    contains
      final :: Cylinder_class_delete
    end type Cylinder_type
  
    interface new
      module procedure Cylinder_class_new
    end interface
    interface setDatum
      module procedure Cylinder_class_setDatum
    end interface
    interface setReferenceFrame
      module procedure Cylinder_class_setReferenceFrame
    end interface
    interface setAlignedCylinder
      module procedure Cylinder_class_setAlignedCylinder
      module procedure Cylinder_class_setAlignedCylinderFlip
    end interface
    interface copy
      module procedure Cylinder_class_copy
    end interface
    interface getDatum
      module procedure Cylinder_class_getDatum
    end interface
    interface getReferenceFrame
      module procedure Cylinder_class_getReferenceFrame
    end interface
    interface getAlignedCylinder
      module procedure Cylinder_class_getAlignedCylinder
    end interface
    interface getCylinderCurvature
      module procedure Cylinder_class_getCurvature
    end interface
    interface getSurfaceArea
      module procedure Cylinder_class_getSurfaceArea
    end interface
    interface printToScreen
      module procedure Cylinder_class_printToScreen
    end interface
  
  
    interface
  
      subroutine F_Cylinder_new(this) &
        bind(C, name="c_Cylinder_new")
        import
        implicit none
        type(c_Cylinder) :: this
      end subroutine F_Cylinder_new
  
      subroutine F_Cylinder_delete(this) &
        bind(C, name="c_Cylinder_delete")
        import
        implicit none
        type(c_Cylinder) :: this
      end subroutine F_Cylinder_delete
  
      subroutine F_Cylinder_setDatum(this, a_datum) &
        bind(C, name="c_Cylinder_setDatum")
        import
        implicit none
        type(c_Cylinder) :: this
        real(C_DOUBLE), dimension(*), intent(in) :: a_datum !  dimension(1:3)
      end subroutine F_Cylinder_setDatum
  
      subroutine F_Cylinder_setReferenceFrame(this, a_normal1, a_normal2, a_normal3) &
        bind(C, name="c_Cylinder_setReferenceFrame")
        import
        implicit none
        type(c_Cylinder) :: this
        real(C_DOUBLE), dimension(*), intent(in) :: a_normal1 !  dimension(1:3)
        real(C_DOUBLE), dimension(*), intent(in) :: a_normal2 !  dimension(1:3)
        real(C_DOUBLE), dimension(*), intent(in) :: a_normal3 !  dimension(1:3)
      end subroutine F_Cylinder_setReferenceFrame
  
      subroutine F_Cylinder_setAlignedCylinder(this, a_radius, a_coeff_b) &
        bind(C, name="c_Cylinder_setAlignedCylinder")
        import
        implicit none
        type(c_Cylinder) :: this
        real(C_DOUBLE), intent(in) :: a_radius 
        real(C_DOUBLE), intent(in) :: a_coeff_b 
      end subroutine F_Cylinder_setAlignedCylinder

      subroutine F_Cylinder_setAlignedCylinderFlip(this, a_radius, a_coeff_b, a_flip) &
        bind(C, name="c_Cylinder_setAlignedCylinderFlip")
        import
        implicit none
        type(c_Cylinder) :: this
        real(C_DOUBLE), intent(in) :: a_radius 
        real(C_DOUBLE), intent(in) :: a_coeff_b 
        real(C_DOUBLE), intent(in) :: a_flip
      end subroutine F_Cylinder_setAlignedCylinderFlip
  
      subroutine F_Cylinder_copy(this, a_other_Cylinder) &
        bind(C, name="c_Cylinder_copy")
        import
        implicit none
        type(c_Cylinder) :: this
        type(c_Cylinder) :: a_other_Cylinder
      end subroutine F_Cylinder_copy
  
      subroutine F_Cylinder_getDatum(this, a_datum) &
        bind(C, name="c_Cylinder_getDatum")
        import
        implicit none
        type(c_Cylinder) :: this
        real(C_DOUBLE), dimension(*), intent(out) :: a_datum
      end subroutine F_Cylinder_getDatum
  
      subroutine F_Cylinder_getReferenceFrame(this, a_frame) &
        bind(C, name="c_Cylinder_getReferenceFrame")
        import
        implicit none
        type(c_Cylinder) :: this
        real(C_DOUBLE), dimension(*), intent(out) :: a_frame
      end subroutine F_Cylinder_getReferenceFrame
  
      subroutine F_Cylinder_getAlignedCylinder(this, a_aligned_Cylinder) &
        bind(C, name="c_Cylinder_getAlignedCylinder")
        import
        implicit none
        type(c_Cylinder) :: this
        real(C_DOUBLE), dimension(*), intent(out) :: a_aligned_Cylinder
      end subroutine F_Cylinder_getAlignedCylinder
  
      function F_Cylinder_getCurvature(this, a_cuboid) result(a_curv) &
        bind(C, name="c_Cylinder_getCurvature")
        import
        implicit none
        type(c_Cylinder) :: this
        type(c_RectCub), intent(in)  :: a_cuboid
        real(C_DOUBLE) :: a_curv
      end function F_Cylinder_getCurvature
  
      function F_Cylinder_getSurfaceArea(this, a_cuboid) result(a_area) &
        bind(C, name="c_Cylinder_getSurfaceArea")
        import
        implicit none
        type(c_Cylinder) :: this
        type(c_RectCub), intent(in)  :: a_cuboid
        real(C_DOUBLE) :: a_area
      end function F_Cylinder_getSurfaceArea
  
      subroutine F_Cylinder_printToScreen(this) &
        bind(C, name="c_Cylinder_printToScreen")
        import
        implicit none
        type(c_Cylinder) :: this
      end subroutine F_Cylinder_printToScreen
  
    end interface
  
  
    contains
  
      subroutine Cylinder_class_new(this)
        implicit none
        type(Cylinder_type), intent(inout) :: this
        call F_Cylinder_new(this%c_object)
      end subroutine Cylinder_class_new
  
      impure elemental subroutine Cylinder_class_delete(this)
        implicit none
        type(Cylinder_type), intent(in) :: this
        call F_Cylinder_delete(this%c_object)
      end subroutine Cylinder_class_delete
  
      subroutine Cylinder_class_setDatum(this, a_datum)
        implicit none
        type(Cylinder_type), intent(in) :: this
        real(IRL_double), dimension(1:3), intent(in) :: a_datum
        call F_Cylinder_setDatum(this%c_object, a_datum)
      end subroutine Cylinder_class_setDatum
  
      subroutine Cylinder_class_setReferenceFrame(this, a_normal1, a_normal2, a_normal3)
        implicit none
        type(Cylinder_type), intent(in) :: this
        real(IRL_double), dimension(1:3), intent(in) :: a_normal1
        real(IRL_double), dimension(1:3), intent(in) :: a_normal2
        real(IRL_double), dimension(1:3), intent(in) :: a_normal3
        call F_Cylinder_setReferenceFrame(this%c_object, a_normal1, a_normal2, a_normal3)
      end subroutine Cylinder_class_setReferenceFrame
  
      subroutine Cylinder_class_setAlignedCylinder(this, a_radius, a_coeff_b)
        implicit none
        type(Cylinder_type), intent(in) :: this
        real(IRL_double), intent(in) :: a_radius
        real(IRL_double), intent(in) :: a_coeff_b
        call F_Cylinder_setAlignedCylinder(this%c_object, a_radius, a_coeff_b)
      end subroutine Cylinder_class_setAlignedCylinder

      subroutine Cylinder_class_setAlignedCylinderFlip(this, a_radius, a_coeff_b, a_flip)
        implicit none
        type(Cylinder_type), intent(in) :: this
        real(IRL_double), intent(in) :: a_radius
        real(IRL_double), intent(in) :: a_coeff_b
        real(IRL_double), intent(in) :: a_flip
        call F_Cylinder_setAlignedCylinderFlip(this%c_object, a_radius, a_coeff_b, a_flip)
      end subroutine Cylinder_class_setAlignedCylinderFlip
  
      subroutine Cylinder_class_copy(this, a_other_Cylinder)
        implicit none
        type(Cylinder_type), intent(inout) :: this
        type(Cylinder_type), intent(in) :: a_other_Cylinder
        call F_Cylinder_copy(this%c_object, a_other_Cylinder%c_object)
      end subroutine Cylinder_class_copy
  
      function Cylinder_class_getDatum(this) result(a_datum)
        implicit none
        type(Cylinder_type), intent(in) :: this
        real(IRL_double), dimension(1:3) :: a_datum
        call F_Cylinder_getDatum(this%c_object, a_datum)
      end function Cylinder_class_getDatum
  
      function Cylinder_class_getReferenceFrame(this) result(a_frame)
        implicit none
        type(Cylinder_type), intent(in) :: this
        real(IRL_double), dimension(1:9) :: a_frame
        call F_Cylinder_getReferenceFrame(this%c_object, a_frame)
      end function Cylinder_class_getReferenceFrame
  
      function Cylinder_class_getAlignedCylinder(this) result(a_aligned_Cylinder)
        implicit none
        type(Cylinder_type), intent(in) :: this
        real(IRL_double), dimension(1:3) :: a_aligned_Cylinder
        call F_Cylinder_getAlignedCylinder(this%c_object, a_aligned_Cylinder)
      end function Cylinder_class_getAlignedCylinder
  
      function Cylinder_class_getCurvature(this, a_cuboid) result(a_curv)
        implicit none
        type(Cylinder_type), intent(in) :: this
        type(RectCub_type), intent(in) :: a_cuboid
        real(IRL_double) :: a_curv
        a_curv = F_Cylinder_getCurvature(this%c_object, a_cuboid%c_object)
        return
      end function Cylinder_class_getCurvature
  
      function Cylinder_class_getSurfaceArea(this, a_cuboid) result(a_area)
        implicit none
        type(Cylinder_type), intent(in) :: this
        type(RectCub_type), intent(in) :: a_cuboid
        real(IRL_double) :: a_area
        a_area = F_Cylinder_getSurfaceArea(this%c_object, a_cuboid%c_object)
        return
      end function Cylinder_class_getSurfaceArea
  
      subroutine Cylinder_class_printToScreen(this)
        implicit none
        type(Cylinder_type), intent(in) :: this
        call F_Cylinder_printToScreen(this%c_object)
      end subroutine Cylinder_class_printToScreen
  
  
  end module f_Cylinder_class
  