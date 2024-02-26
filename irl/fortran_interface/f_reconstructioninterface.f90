!  This file is part of the Interface Reconstruction Library (IRL),
!  a library for interface reconstruction and computational geometry operations.
!
!  Copyright (C) 2019 Robert Chiodi <robert.chiodi@gmail.com>
!
!  This Source Code Form is subject to the terms of the Mozilla Public
!  License, v. 2.0. If a copy of the MPL was not distributed with this
!  file, You can obtain one at https://mozilla.org/MPL/2.0/.

!> \file f_interface_reconstruction.f90
!!
!! This file presents different interface reconstruction
!! methods that can be used to obtain PlanarSeps.

!> This module contains interface reconstruction methods
!! that can be used to obtain PlanarSeps. The
!! requirements to use each type of reconstruction
!! are different. Please consult the documentation
!! and examples before using a specific reconstruction
!! type.
module f_ReconstructionInterface
  use f_DefinedTypes
  use f_SepVM_class
  use f_RectCub_class
  use f_Hex_class  
  use f_Tri_class
  use f_Tet_class
  use f_PlanarSep_class
  use f_ELVIRANeigh_class
  use f_ListVM_VMAN_class
  use f_LVIRANeigh_RectCub_class
  use f_LVIRANeigh_Hex_class
  use f_LVIRANeigh_Tet_class  
  use f_R2PNeigh_RectCub_class
  use f_OptimizationBehavior_class
  use f_R2PWeighting_class
  implicit none

  ! ELVIRA is already a plain name, reconstructELVIRA2D, reconstructELVIRA3D

  interface reconstructMOF2D
    ! 2D MOF reconstruction on a RectCub and default weights
    module procedure reconstructMOF2D_RectCub
    ! 2D MOF reconstruction on a RectCub and given weights
    module procedure reconstructMOF2D_GW_RectCub
    ! 2D MOF reconstruction on a Tri and default weights
    module procedure reconstructMOF2D_Tri
    ! 2D MOF reconstruction on a Tri and given weights
    module procedure reconstructMOF2D_GW_Tri
  end interface reconstructMOF2D

  interface reconstructMOF3D
    ! 3D MOF reconstruction on a RectCub and default weights
    module procedure reconstructMOF3D_RectCub
    ! 3D MOF reconstruction on a RectCub and given weights
    module procedure reconstructMOF3D_GW_RectCub
    ! 3D MOF reconstruction on a Hex and default weights
    module procedure reconstructMOF3D_Hex
    ! 3D MOF reconstruction on a Hex and given weights
    module procedure reconstructMOF3D_GW_Hex
    ! 3D MOF reconstruction on a Tet and default weights
    module procedure reconstructMOF3D_Tet
    ! 3D MOF reconstruction on a Tet and given weights
    module procedure reconstructMOF3D_GW_Tet
  end interface reconstructMOF3D

  interface reconstructAdvectedNormals
    ! Advected normal reconstruction using ListedVolumeMoments<VolumeMomentsAndNormal> 
    ! on a RectCub
    module procedure reconstructAdvectedNormals_RectCub
  end interface reconstructAdvectedNormals

  interface reconstructAdvectedNormalsDbg
    ! Debug version of advected normal reconstruction using
    ! ListedVolumeMoments<VolumeMomentsAndNormal> on a RectCub
    module procedure reconstructAdvectedNormalsDbg_RectCub
  end interface reconstructAdvectedNormalsDbg

  interface reconstructR2P2D
    ! 2D R2P on a RectCub mesh
    module procedure reconstructR2P2D_RectCub
  end interface reconstructR2P2D

  interface reconstructR2P3D
    ! 3D R2P on a RectCub mesh
    module procedure reconstructR2P3D_RectCub
    ! 3D R2P with user-defined weighting on a RectCub mesh
    module procedure reconstructR2P3DwWeights_RectCub
    ! 3D R2P with user-defined optimization parameters and weighting on a RectCub mesh
    module procedure reconstructR2P3DChangeBehavior_RectCub      
  end interface reconstructR2P3D

  interface reconstructR2P2DDbg
    ! Debug version of 2D R2P on a RectCub mesh
    module procedure reconstructR2P2DDbg_RectCub
  end interface reconstructR2P2DDbg

  interface reconstructR2P3DDbg
    ! Debug version of 3D R2P on a RectCub mesh
    module procedure reconstructR2P3DDbg_RectCub
  end interface reconstructR2P3DDbg

  interface reconstructLVIRA2D
    ! 2D LVIRA on a RectCub mesh
    module procedure reconstructLVIRA2D_RectCub
    ! 2D LVIRA on a Hex mesh
    module procedure reconstructLVIRA2D_Hex        
  end interface reconstructLVIRA2D

  interface reconstructLVIRA3D
    ! 3D LVIRA on a RectCub mesh
    module procedure reconstructLVIRA3D_RectCub
    ! 3D LVIRA on a Hex mesh
    module procedure reconstructLVIRA3D_Hex
    ! 3D LVIRA on a Tet mesh
    module procedure reconstructLVIRA3D_Tet        
  end interface reconstructLVIRA3D  

  interface
    subroutine F_reconstructELVIRA2D(a_ELVIRANeigh, a_planar_separator) &
    bind(C, name="c_reconstructELVIRA2D")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_ELVIRANeigh) :: a_ELVIRANeigh ! Pointer to a ELVIRANeigh object
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructELVIRA2D
  end interface

  interface
    subroutine F_reconstructELVIRA3D(a_ELVIRANeigh, a_planar_separator) &
    bind(C, name="c_reconstructELVIRA3D")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_ELVIRANeigh) :: a_ELVIRANeigh ! Pointer to a ELVIRANeigh object
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructELVIRA3D
  end interface

  interface
    subroutine F_reconstructML3(a_LVIRANeigh, a_r2p, a_liquid_centroids, a_gas_centroids, a_planar_separator, flag) &
    bind(C, name="c_reconstructML")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      integer(C_INT), dimension(0:0), intent(in) :: flag
      !type(c_ELVIRANeigh) :: a_ELVIRANeigh
      type(c_LVIRANeigh_RectCub) :: a_LVIRANeigh ! Pointer to a ELVIRANeigh object
      type(c_R2PNeigh_RectCub) :: a_r2p
      real(C_DOUBLE), dimension(0:2,0:26), intent(in) :: a_liquid_centroids
      real(C_DOUBLE), dimension(0:2,0:26), intent(in) :: a_gas_centroids
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructML3
  end interface

  interface
    subroutine F_reconstructML2(a_LVIRANeigh, a_liquid_centroids, a_gas_centroids, a_planar_separator, flag) &
    bind(C, name="c_reconstructML2")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      integer(C_INT), dimension(0:0), intent(in) :: flag
      !type(c_ELVIRANeigh) :: a_ELVIRANeigh
      type(c_LVIRANeigh_RectCub) :: a_LVIRANeigh ! Pointer to a ELVIRANeigh object
      real(C_DOUBLE), dimension(0:2,0:26), intent(in) :: a_liquid_centroids
      real(C_DOUBLE), dimension(0:2,0:26), intent(in) :: a_gas_centroids
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructML2
  end interface

  interface
    subroutine F_reconstructML(normal, vf_center, cell_bound, a_planar_separator) &
    bind(C, name="c_reconstructML")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      real(C_DOUBLE), dimension(0:2), intent(in) :: normal
      real(C_DOUBLE), dimension(0:0), intent(in) :: vf_center
      real(C_DOUBLE), dimension(0:5) :: cell_bound
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructML
  end interface

  interface
    subroutine F_loadML(file) &
    bind(C, name="c_loadML")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      character(kind=c_char) :: file(*)
    end subroutine F_loadML
  end interface

  interface
    subroutine F_loadML2(file) &
    bind(C, name="c_loadML2")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      character(kind=c_char) :: file(*)
    end subroutine F_loadML2
  end interface

  interface
    subroutine F_reconstructMOF2D_RectCub(a_rectangular_cuboid, a_separated_volume_moments, a_planar_separator) &
    bind(C, name="c_reconstructMOF2D_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_RectCub) :: a_rectangular_cuboid ! Pointer to a RectCub object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF2D_RectCub
  end interface

  interface
    subroutine F_reconstructMOF3D_RectCub(a_rectangular_cuboid, a_separated_volume_moments, a_planar_separator) &
    bind(C, name="c_reconstructMOF3D_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_RectCub) :: a_rectangular_cuboid ! Pointer to a RectCub object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF3D_RectCub
  end interface

  interface
    subroutine F_reconstructMOF2D_GW_RectCub(a_rectangular_cuboid, a_separated_volume_moments, &
         a_internal_weight, a_external_weight, a_planar_separator) &
    bind(C, name="c_reconstructMOF2D_GW_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_RectCub) :: a_rectangular_cuboid ! Pointer to a RectCub object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      real(C_DOUBLE), intent(in) :: a_internal_weight ! Assigned weight for internal volume centroid
      real(C_DOUBLE), intent(in) :: a_external_weight ! Assigned weight for external volume centroid
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF2D_GW_RectCub
  end interface

  interface
    subroutine F_reconstructMOF3D_GW_RectCub(a_rectangular_cuboid, a_separated_volume_moments, &
         a_internal_weight, a_external_weight, a_planar_separator) &
    bind(C, name="c_reconstructMOF3D_GW_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_RectCub) :: a_rectangular_cuboid ! Pointer to a RectCub object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      real(C_DOUBLE), intent(in) :: a_internal_weight ! Assigned weight for internal volume centroid
      real(C_DOUBLE), intent(in) :: a_external_weight ! Assigned weight for external volume centroid
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF3D_GW_RectCub
  end interface

  interface
    subroutine F_reconstructMOF3D_Hex(a_hexahedron, a_separated_volume_moments, a_planar_separator) &
    bind(C, name="c_reconstructMOF3D_Hex")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_Hex) :: a_hexahedron ! Pointer to a Hex object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF3D_Hex
  end interface
  
  interface
    subroutine F_reconstructMOF3D_GW_Hex(a_hexahedron, a_separated_volume_moments, &
         a_internal_weight, a_external_weight, a_planar_separator) &
    bind(C, name="c_reconstructMOF3D_GW_Hex")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_Hex) :: a_hexahedron ! Pointer to a Hex object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      real(C_DOUBLE), intent(in) :: a_internal_weight ! Assigned weight for internal volume centroid
      real(C_DOUBLE), intent(in) :: a_external_weight ! Assigned weight for external volume centroid
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF3D_GW_Hex
  end interface
  
  interface
    subroutine F_reconstructMOF2D_Tri(a_tri, a_separated_volume_moments, a_planar_separator) &
    bind(C, name="c_reconstructMOF2D_Tri")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_Tri) :: a_tri ! Pointer to a Tri object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF2D_Tri
  end interface

  interface
    subroutine F_reconstructMOF2D_GW_Tri(a_tri, a_separated_volume_moments, &
         a_internal_weight, a_external_weight, a_planar_separator) &
    bind(C, name="c_reconstructMOF2D_GW_Tri")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_Tri) :: a_tri ! Pointer to a Tri object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      real(C_DOUBLE), intent(in) :: a_internal_weight ! Assigned weight for internal volume centroid
      real(C_DOUBLE), intent(in) :: a_external_weight ! Assigned weight for external volume centroid
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF2D_GW_Tri
  end interface

  interface
    subroutine F_reconstructMOF3D_Tet(a_tet, a_separated_volume_moments, a_planar_separator) &
    bind(C, name="c_reconstructMOF3D_Tet")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_Tet) :: a_tet ! Pointer to a Tet object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF3D_Tet
  end interface

  interface
    subroutine F_reconstructMOF3D_GW_Tet(a_tet, a_separated_volume_moments, &
         a_internal_weight, a_external_weight, a_planar_separator) &
    bind(C, name="c_reconstructMOF3D_GW_Tet")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_Tet) :: a_tet ! Pointer to a Tet object
      type(c_SepVM) :: a_separated_volume_moments ! Pointer to SeparatedMoments<VolumeMoments> object
      real(C_DOUBLE), intent(in) :: a_internal_weight ! Assigned weight for internal volume centroid
      real(C_DOUBLE), intent(in) :: a_external_weight ! Assigned weight for external volume centroid
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructMOF3D_GW_Tet
  end interface

  interface
    subroutine F_reconstructAdvectedNormals_RectCub(a_volume_moments_list, a_neighborhood, &
         a_two_plane_threshold, a_planar_separator) &
    bind(C, name="c_reconstructAdvectedNormals_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_ListVM_VMAN) :: a_volume_moments_list ! Pointer to a ListedVolumeMoments<VolumeMomentsAndNormal>
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      real(C_DOUBLE), intent(in) :: a_two_plane_threshold ! Determines when 1 or 2 plane PlanarSep is created
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructAdvectedNormals_RectCub
  end interface

  interface
    subroutine F_reconstructAdvectedNormalsDbg_RectCub(a_volume_moments_list, a_neighborhood, &
         a_two_plane_threshold, a_planar_separator) &
    bind(C, name="c_reconstructAdvectedNormalsDbg_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_ListVM_VMAN) :: a_volume_moments_list ! Pointer to a ListedVolumeMoments<VolumeMomentsAndNormal>
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      real(C_DOUBLE), intent(in) :: a_two_plane_threshold ! Determines when 1 or 2 plane PlanarSep is created
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep to set
    end subroutine F_reconstructAdvectedNormalsDbg_RectCub
  end interface

  interface
    subroutine F_reconstructR2P2D_RectCub(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructR2P2D_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructR2P2D_RectCub
  end interface

  interface
    subroutine F_reconstructR2P3D_RectCub(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructR2P3D_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructR2P3D_RectCub
  end interface

  interface
    subroutine F_reconstructR2P3DwWeights_RectCub(a_neighborhood, a_planar_separator, a_importances) &
    bind(C, name="c_reconstructR2P3DwWeights_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
      type(c_R2PWeighting) :: a_importances ! Pointer for R2PWeighting
    end subroutine F_reconstructR2P3DwWeights_RectCub
  end interface  

  interface
    subroutine F_reconstructR2P3DChangeBehavior_RectCub(a_neighborhood, a_planar_separator, a_parameters, &
      a_importances) &
    bind(C, name="c_reconstructR2P3DChangeBehavior_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
      type(c_OptimizationBehavior) :: a_parameters ! Pointer for OptimizationBehavior
      type(c_R2PWeighting) :: a_importances ! Pointer for R2PWeighting
    end subroutine F_reconstructR2P3DChangeBehavior_RectCub
  end interface 

  interface
    subroutine F_reconstructR2P2DDbg_RectCub(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructR2P2DDbg_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructR2P2DDbg_RectCub
  end interface

  interface
    subroutine F_reconstructR2P3DDbg_RectCub(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructR2P3DDbg_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_R2PNeigh_RectCub) :: a_neighborhood ! Pointer to a R2PNeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructR2P3DDbg_RectCub
  end interface

  interface
    subroutine F_reconstructLVIRA2D_RectCub(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructLVIRA2D_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_LVIRANeigh_RectCub) :: a_neighborhood ! Pointer to a LVIRANeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructLVIRA2D_RectCub
  end interface

  interface
    subroutine F_reconstructLVIRA3D_RectCub(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructLVIRA3D_RectCub")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_LVIRANeigh_RectCub) :: a_neighborhood ! Pointer to a LVIRANeigh<RectCub>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructLVIRA3D_RectCub
  end interface

  interface
    subroutine F_reconstructLVIRA2D_Hex(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructLVIRA2D_Hex")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_LVIRANeigh_Hex) :: a_neighborhood ! Pointer to a LVIRANeigh<Hex>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructLVIRA2D_Hex
  end interface

  interface
    subroutine F_reconstructLVIRA3D_Hex(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructLVIRA3D_Hex")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_LVIRANeigh_Hex) :: a_neighborhood ! Pointer to a LVIRANeigh<Hex>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructLVIRA3D_Hex
  end interface

  interface
    subroutine F_reconstructLVIRA3D_Tet(a_neighborhood, a_planar_separator) &
    bind(C, name="c_reconstructLVIRA3D_Tet")
      use, intrinsic :: iso_c_binding
      import
      implicit none
      type(c_LVIRANeigh_Tet) :: a_neighborhood ! Pointer to a LVIRANeigh<Tet>
      type(c_PlanarSep) :: a_planar_separator ! Pointer for PlanarSep
    end subroutine F_reconstructLVIRA3D_Tet
  end interface
  
  contains

  subroutine reconstructELVIRA2D(a_elvira_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(ELVIRANeigh_type), intent(in) :: a_elvira_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructELVIRA2D(a_elvira_neighborhood%c_object, a_planar_separator%c_object)

  end subroutine reconstructELVIRA2D

  subroutine reconstructELVIRA3D(a_elvira_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(ELVIRANeigh_type), intent(in) :: a_elvira_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructELVIRA3D(a_elvira_neighborhood%c_object, a_planar_separator%c_object)

  end subroutine reconstructELVIRA3D

  subroutine reconstructML3(a_lvira, a_r2p, a_centroids, a_g_centroids, a_separator, flag)
    use, intrinsic :: iso_c_binding
    implicit none
      integer(c_int), dimension(0:0), intent(in) :: flag
      !type(ELVIRANeigh_type), intent(in) :: a_neighborhood
      type(LVIRANeigh_RectCub_type), intent(in) :: a_lvira
      type(R2PNeigh_RectCub_type), intent(in) :: a_r2p
      real(IRL_double), dimension(0:2,0:26), intent(in) :: a_centroids
      real(IRL_double), dimension(0:2,0:26), intent(in) :: a_g_centroids
      type(PlanarSep_type), intent(inout) :: a_separator
      call F_reconstructML3(a_lvira%c_object, a_r2p%c_object, a_centroids, a_g_centroids, a_separator%c_object, flag)

  end subroutine reconstructML3

  subroutine reconstructML2(a_lvira, a_centroids, a_g_centroids, a_separator, flag)
    use, intrinsic :: iso_c_binding
    implicit none
      integer(c_int), dimension(0:0), intent(in) :: flag
      !type(ELVIRANeigh_type), intent(in) :: a_neighborhood
      type(LVIRANeigh_RectCub_type), intent(in) :: a_lvira
      real(IRL_double), dimension(0:2,0:26), intent(in) :: a_centroids
      real(IRL_double), dimension(0:2,0:26), intent(in) :: a_g_centroids
      type(PlanarSep_type), intent(inout) :: a_separator
      call F_reconstructML2(a_lvira%c_object, a_centroids, a_g_centroids, a_separator%c_object, flag)

  end subroutine reconstructML2

  subroutine reconstructML(normal, vf_center, cell_bound, a_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      real(IRL_DOUBLE), dimension(0:2), intent(in) :: normal
      real(IRL_DOUBLE), dimension(0:0), intent(in) :: vf_center
      real(IRL_DOUBLE), dimension(0:5), intent(in) :: cell_bound
      type(PlanarSep_type), intent(inout) :: a_separator
      call F_reconstructML(normal, vf_center, cell_bound, a_separator%c_object)

  end subroutine reconstructML

  subroutine loadML(name)
    use, intrinsic :: iso_c_binding
    implicit none
      character(kind=c_char), intent(in) :: name(*)
      call F_loadML(name)

  end subroutine loadML

  subroutine loadML2(name)
    use, intrinsic :: iso_c_binding
    implicit none
      character(kind=c_char), intent(in) :: name(*)
      call F_loadML2(name)

  end subroutine loadML2

  subroutine reconstructMOF2D_RectCub(a_rectangular_cuboid, a_separated_volume_moments, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(RectCub_type), intent(in) :: a_rectangular_cuboid
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF2D_RectCub &
          (a_rectangular_cuboid%c_object, a_separated_volume_moments%c_object, a_planar_separator%c_object)

  end subroutine reconstructMOF2D_RectCub

  subroutine reconstructMOF3D_RectCub(a_rectangular_cuboid, a_separated_volume_moments, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(RectCub_type), intent(in) :: a_rectangular_cuboid
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF3D_RectCub &
          (a_rectangular_cuboid%c_object, a_separated_volume_moments%c_object, a_planar_separator%c_object)

  end subroutine reconstructMOF3D_RectCub

  subroutine reconstructMOF2D_GW_RectCub(a_rectangular_cuboid, a_separated_volume_moments, &
       a_internal_weight, a_external_weight, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(RectCub_type), intent(in) :: a_rectangular_cuboid
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      real(IRL_double), intent(in) :: a_internal_weight
      real(IRL_double), intent(in) :: a_external_weight
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF2D_GW_RectCub &
           (a_rectangular_cuboid%c_object, a_separated_volume_moments%c_object, a_internal_weight, &
           a_external_weight, a_planar_separator%c_object)

  end subroutine reconstructMOF2D_GW_RectCub

  subroutine reconstructMOF3D_GW_RectCub(a_rectangular_cuboid, a_separated_volume_moments, &
       a_internal_weight, a_external_weight, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(RectCub_type), intent(in) :: a_rectangular_cuboid
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      real(IRL_double), intent(in) :: a_internal_weight
      real(IRL_double), intent(in) :: a_external_weight
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF3D_GW_RectCub &
           (a_rectangular_cuboid%c_object, a_separated_volume_moments%c_object, &
           a_internal_weight, a_external_weight, a_planar_separator%c_object)

  end subroutine reconstructMOF3D_GW_RectCub

  subroutine reconstructMOF3D_Hex(a_hexahedron, a_separated_volume_moments, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(Hex_type), intent(in) :: a_hexahedron
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF3D_Hex &
          (a_hexahedron%c_object, a_separated_volume_moments%c_object, a_planar_separator%c_object)

  end subroutine reconstructMOF3D_Hex

  subroutine reconstructMOF3D_GW_Hex(a_hexahedron, a_separated_volume_moments, &
       a_internal_weight, a_external_weight, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(Hex_type), intent(in) :: a_hexahedron
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      real(IRL_double), intent(in) :: a_internal_weight
      real(IRL_double), intent(in) :: a_external_weight
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF3D_GW_Hex &
           (a_hexahedron%c_object, a_separated_volume_moments%c_object, &
           a_internal_weight, a_external_weight, a_planar_separator%c_object)

  end subroutine reconstructMOF3D_GW_Hex
  
  subroutine reconstructMOF2D_Tri(a_tri, a_separated_volume_moments, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(Tri_type), intent(in) :: a_tri
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF2D_Tri &
          (a_tri%c_object, a_separated_volume_moments%c_object, a_planar_separator%c_object)

  end subroutine reconstructMOF2D_Tri

  subroutine reconstructMOF2D_GW_Tri(a_tri, a_separated_volume_moments, &
       a_internal_weight, a_external_weight, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(Tri_type), intent(in) :: a_tri
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      real(IRL_double), intent(in) :: a_internal_weight
      real(IRL_double), intent(in) :: a_external_weight
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF2D_GW_Tri &
           (a_tri%c_object, a_separated_volume_moments%c_object, &
           a_internal_weight, a_external_weight, a_planar_separator%c_object)

  end subroutine reconstructMOF2D_GW_Tri

  subroutine reconstructMOF3D_Tet(a_tet, a_separated_volume_moments, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(Tet_type), intent(in) :: a_tet
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF3D_Tet &
          (a_tet%c_object, a_separated_volume_moments%c_object, a_planar_separator%c_object)

  end subroutine reconstructMOF3D_Tet

  subroutine reconstructMOF3D_GW_Tet(a_tet, a_separated_volume_moments, &
       a_internal_weight, a_external_weight, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(Tet_type), intent(in) :: a_tet
      type(SepVM_type), intent(in) :: a_separated_volume_moments
      real(IRL_double), intent(in) :: a_internal_weight
      real(IRL_double), intent(in) :: a_external_weight
      type(PlanarSep_type), intent(inout) :: a_planar_separator

      call F_reconstructMOF3D_GW_Tet &
           (a_tet%c_object, a_separated_volume_moments%c_object, &
           a_internal_weight, a_external_weight, a_planar_separator%c_object)

  end subroutine reconstructMOF3D_GW_Tet

  subroutine reconstructAdvectedNormals_RectCub(a_volume_moments_list, a_neighborhood, &
       a_two_plane_threshold, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(ListVM_VMAN_type), intent(in) :: a_volume_moments_list
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      real(IRL_double), intent(in) :: a_two_plane_threshold
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructAdvectedNormals_RectCub(a_volume_moments_list%c_object, &
           a_neighborhood%c_object, a_two_plane_threshold, a_planar_separator%c_object)
  end subroutine reconstructAdvectedNormals_RectCub

  subroutine reconstructAdvectedNormalsDbg_RectCub(a_volume_moments_list, a_neighborhood, a_two_plane_threshold, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(ListVM_VMAN_type), intent(in) :: a_volume_moments_list
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      real(IRL_double), intent(in) :: a_two_plane_threshold
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructAdvectedNormalsDbg_RectCub(a_volume_moments_list%c_object, &
           a_neighborhood%c_object, a_two_plane_threshold, a_planar_separator%c_object)
  end subroutine reconstructAdvectedNormalsDbg_RectCub

  subroutine reconstructR2P2D_RectCub(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructR2P2D_RectCub(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructR2P2D_RectCub

  subroutine reconstructR2P3D_RectCub(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructR2P3D_RectCub(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructR2P3D_RectCub

  subroutine reconstructR2P3DwWeights_RectCub(a_neighborhood, a_planar_separator, a_importances)
    use, intrinsic :: iso_c_binding
    implicit none
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      type(R2PWeighting_type), intent(in) :: a_importances
      call F_reconstructR2P3DwWeights_RectCub(a_neighborhood%c_object, a_planar_separator%c_object, a_importances%c_object)
  end subroutine reconstructR2P3DwWeights_RectCub  

  subroutine reconstructR2P3DChangeBehavior_RectCub(a_neighborhood, a_planar_separator, a_parameters, &
    a_importances)
    use, intrinsic :: iso_c_binding
    implicit none
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      type(OptimizationBehavior_type), intent(in) :: a_parameters
      type(R2PWeighting_type), intent(in) :: a_importances
      call F_reconstructR2P3DChangeBehavior_RectCub(a_neighborhood%c_object, a_planar_separator%c_object, a_parameters%c_object, &
      a_importances%c_object)
  end subroutine reconstructR2P3DChangeBehavior_RectCub  

  subroutine reconstructR2P2DDbg_RectCub(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructR2P2DDbg_RectCub(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructR2P2DDbg_RectCub

  subroutine reconstructR2P3DDbg_RectCub(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(R2PNeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructR2P3DDbg_RectCub(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructR2P3DDbg_RectCub

  subroutine reconstructLVIRA2D_RectCub(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(LVIRANeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructLVIRA2D_RectCub(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructLVIRA2D_RectCub

  subroutine reconstructLVIRA3D_RectCub(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(LVIRANeigh_RectCub_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructLVIRA3D_RectCub(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructLVIRA3D_RectCub

  subroutine reconstructLVIRA2D_Hex(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(LVIRANeigh_Hex_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructLVIRA2D_Hex(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructLVIRA2D_Hex

  subroutine reconstructLVIRA3D_Hex(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(LVIRANeigh_Hex_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructLVIRA3D_Hex(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructLVIRA3D_Hex

  subroutine reconstructLVIRA3D_Tet(a_neighborhood, a_planar_separator)
    use, intrinsic :: iso_c_binding
    implicit none
      type(LVIRANeigh_Tet_type), intent(in) :: a_neighborhood
      type(PlanarSep_type), intent(inout) :: a_planar_separator
      call F_reconstructLVIRA3D_Tet(a_neighborhood%c_object, a_planar_separator%c_object)
  end subroutine reconstructLVIRA3D_Tet 
  
end module f_ReconstructionInterface
