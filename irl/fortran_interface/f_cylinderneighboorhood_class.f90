!  This file is part of the Interface Reconstruction Library (IRL),
!  a library for interface reconstruction and computational geometry operations.
!
!  Copyright (C) 2025 Andrew Cahaly <andrew.cahaly@gmail.com>
!
!  This Source Code Form is subject to the terms of the Mozilla Public
!  License, v. 2.0. If a copy of the MPL was not distributed with this
!  file, You can obtain one at https://mozilla.org/MPL/2.0/.

!> \file f_cylinderneighborhood.f90
!!
!! This file contains functions reproducing
!! the functionality of the IRL class
!! cylinderNeighborhood. The purpose of this
!! is to allow building the stencil
!! through references to then supply
!! to obtain a cylinder.

!> \brief A fortran type class to 
!! provide the functionality of 
!! cylinderNeighborhood.
module f_cylinderNeigh_class
    use f_RectCub_class
    use, intrinsic :: iso_c_binding
    use f_DefinedTypes
    use f_VM_class
    implicit none
  
    type, public, bind(C) :: c_cylinderNeigh
      type(C_PTR), private :: object = C_NULL_PTR
    end type c_cylinderNeigh
  
    type, public :: cylinderNeigh_type
      type(c_cylinderNeigh) :: c_object
    contains
      final :: cylinderNeigh_class_delete
    end type cylinderNeigh_type
  
    interface new
      module procedure cylinderNeigh_class_new
    end interface
    interface setSize
      module procedure cylinderNeigh_class_setSize
    end interface
    interface setMember
      module procedure cylinderNeigh_class_setMember
    end interface
  
    interface
  
      subroutine F_cylinderNeigh_new(this) &
        bind(C, name="c_cylinderNeigh_new")
        import
        implicit none
        type(c_cylinderNeigh) :: this
      end subroutine F_cylinderNeigh_new
  
      subroutine F_cylinderNeigh_delete(this) &
        bind(C, name="c_cylinderNeigh_delete")
        import
        implicit none
        type(c_cylinderNeigh) :: this
      end subroutine F_cylinderNeigh_delete
  
      subroutine F_cylinderNeigh_setSize(this, a_size) &
        bind(C, name="c_cylinderNeigh_setSize")
        import
        implicit none
        type(c_cylinderNeigh) :: this
        integer(C_INT) :: a_size
      end subroutine F_cylinderNeigh_setSize
  
      subroutine F_cylinderNeigh_setMember(this, a_rectangular_cuboid, &
          a_volume_moments, i, j, k) &
        bind(C, name="c_cylinderNeigh_setMember")
        import
        implicit none
        type(c_cylinderNeigh) :: this
        type(c_RectCub) :: a_rectangular_cuboid ! Pointer to RectCub
        type(c_VM), intent(in) :: a_volume_moments
        integer(C_INT), intent(in) :: i
        integer(C_INT), intent(in) :: j
        integer(C_INT), intent(in) :: k
      end subroutine F_cylinderNeigh_setMember
  
    end interface
  
  
    contains
  
      subroutine cylinderNeigh_class_new(this)
        implicit none
        type(cylinderNeigh_type), intent(inout) :: this
        call F_cylinderNeigh_new(this%c_object)
      end subroutine cylinderNeigh_class_new
  
      impure elemental subroutine cylinderNeigh_class_delete(this)
        implicit none
        type(cylinderNeigh_type), intent(in) :: this
        call F_cylinderNeigh_delete(this%c_object)
      end subroutine cylinderNeigh_class_delete
  
      subroutine cylinderNeigh_class_setSize(this, a_size)
        implicit none
        type(cylinderNeigh_type), intent(in) :: this
        integer(IRL_UnsignedIndex_t), intent(in) :: a_size
        call F_cylinderNeigh_setSize(this%c_object,a_size)
      end subroutine cylinderNeigh_class_setSize
  
      subroutine cylinderNeigh_class_setMember(this, a_rectangular_cuboid, &
            a_volume_moments, i, j, k)
        implicit none
        type(cylinderNeigh_type), intent(in) :: this
        type(RectCub_type), intent(in) :: a_rectangular_cuboid
        type(VM_type), intent(in) :: a_volume_moments
        integer(IRL_SignedIndex_t), intent(in) :: i
        integer(IRL_SignedIndex_t), intent(in) :: j
        integer(IRL_SignedIndex_t), intent(in) :: k
        call F_cylinderNeigh_setMember(this%c_object,a_rectangular_cuboid%c_object, &
            a_volume_moments%c_object,i,j,k)
      end subroutine cylinderNeigh_class_setMember
  
  end module f_cylinderNeigh_class
  