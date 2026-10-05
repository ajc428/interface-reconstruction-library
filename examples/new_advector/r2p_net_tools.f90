!> Helpers for the R2P-Net reconstruction in vfs_class (build_r2p_net and
!> r2p_paraboloid). Fortran ports of the C++ pipeline in examples/new_advector:
!>
!>   r2p_phase_is_gas        phase fed to the network (r2pFlip, reconstruction_types.cpp)
!>   r2p_classifier_stencil  the 5^3 ml_classifier input R2P3D_Net builds
!>   r2p_newton_distances    two-plane distance solve (R2PNewtonDistanceSolver,
!>                           r2p_newton_distance.h)
!>   r2p_clean               IRL::cleanReconstruction
!>   r2p_snap_thin_film      thin-film snap to a slab (r2p_snap.h), with
!>   r2p_is_slab, r2p_keep_slab   its pass-2 counterparts
!>   r2p_tip_spread          film-tip sensor (r2p_tip_sensor.h filmSpread; NGA2 detect_lig_edge)
!>   r2p_edge_count          film-edge sensor (r2p_edge_sensor.h; NGA2 detect_edge_regions),
!>   r2p_mean_normal         with the network's mean normal for its tangent plane
!>   r2p_edge_topology       topological film-edge sensor (r2p_edge_topology.h), and
!>   r2p_is_edge             which of the two sensors gates pinch prevention (not where the guard holds)
!>   r2p_film_guard          thin-film guard for the R2P routing (r2p_edge_topology.h filmSeparates)
!>   r2p_prevent_pinch       open two planes that would pinch a continuing film (r2p_nopinch.h)
!>   r2p_planes, r2p_planes_cross, r2p_nopinch_report   its self-check (r2p_nopinch_debug)
!>
!> Everything takes explicit arguments, so the module also builds and tests
!> outside NGA2 against IRL's Fortran interface. r2p_newton_distances takes a
!> PlanarSep or a SeparatorVariant (NGA2's liquid_gas_interface).
module r2p_net_tools
   use precision, only: WP
   use irl_fortran_interface
   implicit none
   private
   public :: r2p_phase_is_gas,r2p_classifier_stencil,r2p_newton_distances,r2p_clean
   public :: r2p_snap_thin_film,r2p_is_slab,r2p_keep_slab,r2p_guard_slab,r2p_pca_slab,r2p_film_thickness,r2p_stencil_thickness,r2p_very_thin_stencil,r2p_tip_spread,r2p_edge_count,r2p_mean_normal
   public :: r2p_edge_topology,r2p_is_edge,r2p_film_guard
   public :: r2p_prevent_pinch
   public :: r2p_planes,r2p_planes_cross,r2p_nopinch_report

   interface r2p_prevent_pinch
      module procedure r2p_prevent_pinch_planar,r2p_prevent_pinch_variant
   end interface r2p_prevent_pinch

   ! Edge sensor: this many empty probe cells or more is an edge (planes may meet)
   real(WP), parameter, public :: r2p_edge_min_count=2.0_WP
   ! Pinch prevention skips the edges of the topological sensor (r2p_edge_topology)
   ! if true, of the presence sensor (r2p_edge_count) if false
   logical , parameter, public :: r2p_edge_use_topology=.true.
   ! Pinch prevention: the film may be nowhere in the cell thinner than this
   ! fraction of its mean thickness (as r2p_nopinch.h; < 0: off)
   real(WP), parameter, public :: r2p_nopinch_gap_fraction=0.1_WP
   ! Self-check of pinch prevention: vfs reports (stdout, lines starting with
   ! R2P_NOPINCH_DEBUG) every non-edge two-plane cell whose planes still cross
   ! inside it after either pass, or at the end of the reconstruction, with
   ! what is needed to replay it offline; at most r2p_nopinch_debug_max
   ! reports per process. Costs an extra 22 values per cell while on.
   logical , parameter, public :: r2p_nopinch_debug=.true.
   integer , parameter :: r2p_nopinch_debug_max=50
   ! Diagnostic dump (vfs build_r2p_net): every r2p_plic_dump_every-th
   ! reconstruction, each interface cell that ended with one plane is appended
   ! to r2p_plic_cells_<rank>.txt (format there; C++: R2P_PLIC_DUMP)
   logical , parameter, public :: r2p_plic_dump=.true.
   integer , parameter, public :: r2p_plic_dump_every=1
   integer , save :: r2p_nopinch_reports=0

   ! Thin-film snap: off by default (thin films have not been seen to break
   ! without it); thresholds as in r2p_snap.h
   logical , parameter, public :: r2p_snap_enabled=.false.
   real(WP), parameter, public :: r2p_snap_max_thickness=0.05_WP     !< film thickness, in cell widths
   real(WP), parameter, public :: r2p_snap_max_opening_deg=1.0_WP    !< angle between the two faces
   integer , parameter, public :: r2p_sheet_end_class=6              !< ml_classifier id, never snapped
   ! Very thin film: PCA slab (r2p_pca_slab) where the film in the 3^3 stencil
   ! is at most this thick, in cell widths (0.005: the thinnest films in the
   ! training data); 0 turns it off
   real(WP), parameter, public :: r2p_pca_slab_max_thickness=0.005_WP

   interface r2p_newton_distances
      module procedure r2p_newton_distances_planar,r2p_newton_distances_variant
   end interface r2p_newton_distances

   ! Scratch IRL objects, allocated once rather than per cell (whether a
   ! compiler finalizes local IRL objects on return has varied)
   logical, save :: ws_ready=.false.
   type(PlanarSep_type), save :: ws_best,ws_start,ws_sep,ws_mid,ws_np
   type(SepVM_type)    , save :: ws_svm
   type(Poly_type)     , save :: ws_poly
   type(RectCub_type)  , save :: ws_block

contains

   subroutine init_workspace()
      implicit none
      if (ws_ready) return
      call new(ws_best); call new(ws_start); call new(ws_sep); call new(ws_mid); call new(ws_np); call new(ws_svm); call new(ws_poly)
      call new(ws_block)
      ws_ready=.true.
   end subroutine init_workspace

   !> Phase 0 for the classifier and R2P-Net (true = gas). The network was
   !> trained with phase 0 = the film, the phase between the two interfaces. A
   !> film's centroids form a flat layer while the surrounding phase fills the
   !> stencil on both sides, so the film is the flatter cloud (smaller ratio of
   !> smallest to largest covariance eigenvalue). Falls back on the VF-sum rule
   !> (phase 0 = minority phase) when either phase is in fewer than 3 cells.
   logical function r2p_phase_is_gas(vf,lbary,gbary,vflo) result(gas)
      implicit none
      real(WP), dimension(-1:1,-1:1,-1:1),     intent(in) :: vf
      real(WP), dimension(3,-1:1,-1:1,-1:1),   intent(in) :: lbary,gbary
      real(WP), intent(in) :: vflo
      real(WP) :: s_liq,s_gas
      s_liq=sphericity(.false.)
      s_gas=sphericity(.true.)
      if (s_liq.lt.0.0_WP.or.s_gas.lt.0.0_WP) then
         gas=(sum(vf).ge.0.5_WP*27.0_WP)
      else
         gas=(s_gas.lt.s_liq)
      end if
   contains
      real(WP) function sphericity(of_gas) result(s)
         implicit none
         logical, intent(in) :: of_gas
         real(WP), dimension(3,27) :: pts
         real(WP), dimension(3,3) :: cov
         real(WP), dimension(3) :: mean,d,ev
         real(WP), dimension(64) :: work
         real(WP) :: f
         integer :: ii,jj,kk,n,m,info
         n=0
         do kk=-1,1; do jj=-1,1; do ii=-1,1
            f=vf(ii,jj,kk); if (of_gas) f=1.0_WP-f
            if (f.le.vflo) cycle
            n=n+1
            if (of_gas) then; pts(:,n)=gbary(:,ii,jj,kk); else; pts(:,n)=lbary(:,ii,jj,kk); end if
         end do; end do; end do
         s=-1.0_WP
         if (n.lt.3) return
         mean=sum(pts(:,1:n),dim=2)/real(n,WP)
         cov=0.0_WP
         do m=1,n
            d=pts(:,m)-mean
            cov(:,1)=cov(:,1)+d*d(1); cov(:,2)=cov(:,2)+d*d(2); cov(:,3)=cov(:,3)+d*d(3)
         end do
         call dsyev('N','U',3,cov,3,ev,work,64,info)
         if (info.ne.0.or.ev(3).le.1.0e-30_WP) return
         s=max(0.0_WP,ev(1))/ev(3)
      end function sphericity
   end function r2p_phase_is_gas

   !> The 5^3 ml_classifier input around a cell, exactly as R2P3D_Net builds it:
   !> phase-0 VF, and phase-0 barycenters relative to the CENTER cell, scaled by
   !> its size and weighted by the VF. Index (a,b,c) = offset (a-3,b-3,c-3).
   subroutine r2p_classifier_stencil(vf,lbary,gbary,center,dxyz,gas,vfrac,bary)
      implicit none
      real(WP), dimension(-2:2,-2:2,-2:2),   intent(in) :: vf
      real(WP), dimension(3,-2:2,-2:2,-2:2), intent(in) :: lbary,gbary
      real(WP), dimension(3), intent(in) :: center,dxyz
      logical, intent(in) :: gas
      real(WP), dimension(5,5,5),   intent(out) :: vfrac
      real(WP), dimension(5,5,5,3), intent(out) :: bary
      real(WP), dimension(3) :: b
      real(WP) :: f
      integer :: ii,jj,kk
      do kk=-2,2; do jj=-2,2; do ii=-2,2
         f=vf(ii,jj,kk)
         if (gas) then
            f=1.0_WP-f
            b=gbary(:,ii,jj,kk)
         else
            b=lbary(:,ii,jj,kk)
         end if
         vfrac(ii+3,jj+3,kk+3)=f
         bary(ii+3,jj+3,kk+3,:)=f*(b-center)/dxyz
      end do; end do; end do
   end subroutine r2p_classifier_stencil

   !> Two-plane distance solve matching the volume fraction AND the first
   !> moment across the film. The separator must hold the two (fixed) normals
   !> and the flip state; both distances are overwritten.
   !>
   !> Newton on (d0,d1) for r = (V/vol - vf, m.(M - M_target)/(vol L)), with
   !> m = n0 - n1 across the film and M the liquid first moment. The target
   !> comes from the FILM phase's own centroid (gas when flipped), so a thin gas
   !> film does not recover it from the liquid centroid, which would amplify any
   !> liquid/gas moment inconsistency by vf/(1-vf). Jacobian from the interface
   !> polygons: moving plane i sweeps its polygon, so dV/dd_i = A_i and
   !> dM/dd_i = A_i p_i (both flip states). Volume is always conserved: the
   !> result is finished with a rigid shift of both planes that matches vf.
   subroutine r2p_newton_distances_planar(cell,sep,vf,lbary,gbary)
      implicit none
      type(RectCub_type),   intent(inout) :: cell
      type(PlanarSep_type), intent(inout) :: sep
      real(WP), intent(in) :: vf
      real(WP), dimension(3), intent(in) :: lbary,gbary
      real(WP), parameter :: vf_tol=1.0e-13_WP
      real(WP), dimension(4) :: plane
      real(WP), dimension(3) :: n0,n1,m,cc,film,lo,hi
      real(WP) :: vol,L,m_target,r0,r1,merit,best_merit,det,dd0,dd1,scale,area,v,first
      real(WP), dimension(0:1) :: A,Ap,dist
      logical :: flipped
      integer :: it,p

      if (getNumberOfPlanes(sep).ne.2) then
         call matchVolumeFraction(cell,vf,sep,vf_tol)
         return
      end if
      call init_workspace()

      call getBoundingPts(cell,lo,hi)
      cc=0.5_WP*(lo+hi); vol=product(hi-lo); L=vol**(1.0_WP/3.0_WP)
      plane=getPlane(sep,0); n0=plane(1:3)/norm2(plane(1:3))
      plane=getPlane(sep,1); n1=plane(1:3)/norm2(plane(1:3))
      m=n0-n1
      if (norm2(m).lt.1.0e-12_WP) m=n0
      m=m/norm2(m)

      ! Initial guess: both planes through the film centroid, shifted rigidly
      ! to match volume
      flipped=isFlipped(sep)
      film=lbary; if (flipped) film=gbary
      call setPlane(sep,0,n0,dot_product(n0,film))
      call setPlane(sep,1,n1,dot_product(n1,film))
      call r2p_match_volume(cell,vf,sep,vf_tol)

      if (flipped) then
         m_target=dot_product(m,cc)-(1.0_WP-vf)*dot_product(m,gbary)
      else
         m_target=vf*dot_product(m,lbary)
      end if

      call copy(ws_best,sep)
      best_merit=huge(1.0_WP)
      do it=1,20
         call getNormMoments(cell,sep,ws_svm)
         v=getVolume(ws_svm,0)
         r0=v/vol-vf
         first=0.0_WP; if (v.gt.0.0_WP) first=v*dot_product(m,getCentroid(ws_svm,0))
         r1=(first/vol-m_target)/L
         merit=r0*r0+r1*r1
         if (merit.lt.best_merit) then
            best_merit=merit
            call copy(ws_best,sep)
         end if
         if (abs(r0).lt.vf_tol.and.abs(r1).lt.vf_tol) exit

         ! Polygon area and m-first-moment, per plane
         do p=0,1
            call getPoly(cell,sep,p,ws_poly)
            A(p)=0.0_WP; Ap(p)=0.0_WP
            if (getNumberOfVertices(ws_poly).lt.3) cycle
            area=abs(calculateVolume(ws_poly))
            A(p)=area/vol
            Ap(p)=area*dot_product(m,calculateCentroid(ws_poly))/(vol*L)
         end do
         det=A(0)*Ap(1)-A(1)*Ap(0)
         if (abs(det).lt.1.0e-14_WP) exit   ! one plane has left the cell
         dd0=-(Ap(1)*r0-A(1)*r1)/det
         dd1=-(A(0)*r1-Ap(0)*r0)/det
         ! Residuals are only piecewise smooth: keep a step within one cell size
         scale=max(1.0_WP,max(abs(dd0),abs(dd1))/L)
         plane=getPlane(sep,0); dist(0)=plane(4)+dd0/scale
         plane=getPlane(sep,1); dist(1)=plane(4)+dd1/scale
         call setPlane(sep,0,n0,dist(0))
         call setPlane(sep,1,n1,dist(1))
      end do

      call copy(sep,ws_best)
      call r2p_match_volume(cell,vf,sep,vf_tol)
      call r2p_clean(cell,vf,sep)
   end subroutine r2p_newton_distances_planar

   !> r2p_newton_distances on a SeparatorVariant: planes and flip are copied to
   !> a PlanarSep, solved there, and copied back
   subroutine r2p_newton_distances_variant(cell,sep,vf,lbary,gbary)
      implicit none
      type(RectCub_type),          intent(inout) :: cell
      type(SeparatorVariant_type), intent(inout) :: sep
      real(WP), intent(in) :: vf
      real(WP), dimension(3), intent(in) :: lbary,gbary
      real(WP), dimension(4) :: plane
      integer :: p
      call init_workspace()
      call setNumberOfPlanes(ws_sep,getNumberOfPlanes(sep))
      do p=0,getNumberOfPlanes(sep)-1
         plane=getPlane(sep,p)
         call setPlane(ws_sep,p,plane(1:3),plane(4))
      end do
      call setFlip(ws_sep,logical(isFlipped(sep)))
      call r2p_newton_distances_planar(cell,ws_sep,vf,lbary,gbary)
      call setNumberOfPlanes(sep,getNumberOfPlanes(ws_sep))
      do p=0,getNumberOfPlanes(ws_sep)-1
         plane=getPlane(ws_sep,p)
         call setPlane(sep,p,plane(1:3),plane(4))
      end do
      call setFlip(sep,logical(isFlipped(ws_sep)))
   end subroutine r2p_newton_distances_variant

   !> Rigid shift of all planes to match vf. IRL's solver first; it can miss on
   !> near-empty/near-full cells (VF ~ 1e-6), so bisect on the shift if it does.
   !> Liquid volume grows monotonically with the shift in both flip states.
   subroutine r2p_match_volume(cell,vf,sep,tol)
      implicit none
      type(RectCub_type),   intent(inout) :: cell
      type(PlanarSep_type), intent(inout) :: sep
      real(WP), intent(in) :: vf,tol
      real(WP), dimension(4) :: plane
      real(WP), dimension(3) :: lo,hi
      real(WP) :: vol,a,b,mid,e
      integer :: it
      call init_workspace()
      call getBoundingPts(cell,lo,hi); vol=product(hi-lo)
      call copy(ws_start,sep)
      call matchVolumeFraction(cell,vf,sep,tol)
      if (abs(vf_error(sep)).le.tol) return
      a=-vol**(1.0_WP/3.0_WP); b=-a
      do while (vf_error_shifted(a).gt.0.0_WP); a=2.0_WP*a; end do
      do while (vf_error_shifted(b).lt.0.0_WP); b=2.0_WP*b; end do
      do it=1,200
         if (b-a.le.epsilon(1.0_WP)*(abs(a)+abs(b))) exit
         mid=0.5_WP*(a+b)
         e=vf_error_shifted(mid)
         if (abs(e).le.tol) exit
         if (e.lt.0.0_WP) then; a=mid; else; b=mid; end if
      end do
   contains
      real(WP) function vf_error(s) result(err)
         type(PlanarSep_type), intent(inout) :: s
         call getNormMoments(cell,s,ws_svm)
         err=getVolume(ws_svm,0)/vol-vf
      end function vf_error
      !> Leaves sep at the shifted state
      real(WP) function vf_error_shifted(t) result(err)
         real(WP), intent(in) :: t
         integer :: p
         call copy(sep,ws_start)
         do p=0,getNumberOfPlanes(ws_start)-1
            plane=getPlane(ws_start,p)
            call setPlane(sep,p,plane(1:3),plane(4)+t)
         end do
         err=vf_error(sep)
      end function vf_error_shifted
   end subroutine r2p_match_volume

   !> IRL::cleanReconstruction: drop planes that do not cut the cell (all 8
   !> corners on one side; a corner counts as above when its signed distance is
   !> > 0), a pure-phase plane if none remain, and merge two identical planes.
   subroutine r2p_clean(cell,vf,sep)
      implicit none
      type(RectCub_type),   intent(inout) :: cell
      type(PlanarSep_type), intent(inout) :: sep
      real(WP), intent(in) :: vf
      real(WP), dimension(4) :: p0,p1,plane
      real(WP), dimension(3) :: lo,hi,x
      integer :: n,c,above,np
      call getBoundingPts(cell,lo,hi)
      do n=getNumberOfPlanes(sep)-1,0,-1
         plane=getPlane(sep,n)
         above=0
         do c=0,7
            x=[merge(hi(1),lo(1),btest(c,0)),merge(hi(2),lo(2),btest(c,1)),merge(hi(3),lo(3),btest(c,2))]
            if (dot_product(plane(1:3),x)-plane(4).gt.0.0_WP) above=above+1
         end do
         if (above.gt.0.and.above.lt.8) cycle
         ! Remove plane n, keeping the order of the others
         np=getNumberOfPlanes(sep)
         if (n.lt.np-1) then
            p1=getPlane(sep,n+1)
            call setPlane(sep,n,p1(1:3),p1(4))
         end if
         call setNumberOfPlanes(sep,np-1)
      end do
      if (getNumberOfPlanes(sep).eq.0) then
         call setNumberOfPlanes(sep,1)
         call setFlip(sep,.false.)
         call setPlane(sep,0,[0.0_WP,0.0_WP,0.0_WP],sign(1.0_WP,vf-0.5_WP))
         return
      end if
      if (getNumberOfPlanes(sep).eq.2) then
         p0=getPlane(sep,0); p1=getPlane(sep,1)
         if (all(p0.eq.p1)) then
            call setNumberOfPlanes(sep,1)
            call matchVolumeFraction(cell,vf,sep)
         end if
      end if
   end subroutine r2p_clean

   !> Thin-film snap (C++: r2p_snap.h). In a film thinner than a few hundredths
   !> of a cell, R2P-Net's small error in the angle between its two faces makes
   !> the planes pinch or cross inside the cell. When the cell is not a sheet
   !> end, the faces open by at most r2p_snap_max_opening_deg, and the film is
   !> at most r2p_snap_max_thickness cells thick, both normals become
   !> +-normalize(n0-n1), a slab. Thickness = film volume / area of the plane
   !> through the film barycenter along that mean normal, clipped to the cell,
   !> in cell widths. n0,n1: unit face normals (either sign, faces opposed);
   !> film_vf, film_bary: of the film phase (gas when the film is gas); cls: the
   !> classifier id (0 when unknown). Returns true if the normals were snapped.
   !> Off unless r2p_snap_enabled.
   logical function r2p_snap_thin_film(cell,film_vf,film_bary,cls,n0,n1) result(snapped)
      implicit none
      type(RectCub_type), intent(inout) :: cell
      real(WP), intent(in) :: film_vf
      real(WP), dimension(3), intent(in) :: film_bary
      integer, intent(in) :: cls
      real(WP), dimension(3), intent(inout) :: n0,n1
      real(WP), dimension(3) :: avg
      real(WP) :: opening
      snapped=.false.
      if (.not.r2p_snap_enabled) return
      if (cls.eq.r2p_sheet_end_class) return
      opening=acos(max(-1.0_WP,min(1.0_WP,-dot_product(n0,n1))))
      if (opening.gt.r2p_snap_max_opening_deg*acos(-1.0_WP)/180.0_WP) return
      avg=n0-n1
      if (norm2(avg).le.0.0_WP) return
      avg=avg/norm2(avg)
      if (r2p_film_thickness(cell,film_vf,film_bary,avg).gt.r2p_snap_max_thickness) return
      n0=avg; n1=-avg
      snapped=.true.
   end function r2p_snap_thin_film

   !> Film thickness in cell widths (C++: r2p_snap.h filmThickness): film volume
   !> over the area of the plane through the film barycenter with normal n,
   !> clipped to the cell; huge if that plane misses the cell.
   real(WP) function r2p_film_thickness(cell,film_vf,film_bary,n) result(t)
      implicit none
      type(RectCub_type), intent(inout) :: cell
      real(WP), intent(in) :: film_vf
      real(WP), dimension(3), intent(in) :: film_bary,n
      real(WP), dimension(3) :: lo,hi
      real(WP) :: area,vol
      t=huge(1.0_WP)
      call init_workspace()
      call getBoundingPts(cell,lo,hi); vol=product(hi-lo)
      call setNumberOfPlanes(ws_mid,1)
      call setPlane(ws_mid,0,n,dot_product(n,film_bary))
      call setFlip(ws_mid,.false.)
      call getPoly(cell,ws_mid,0,ws_poly)
      if (getNumberOfVertices(ws_poly).lt.3) return
      area=abs(calculateVolume(ws_poly))
      if (area.le.1.0e-12_WP*vol**(2.0_WP/3.0_WP)) return
      t=film_vf*vol/area/vol**(1.0_WP/3.0_WP)
   end function r2p_film_thickness

   !> Film thickness over the 3^3 stencil, in cell widths (C++: r2p_snap.h
   !> stencilThickness): the film phase's volume in the stencil over the area of
   !> the plane with normal n through its centroid, clipped to the stencil.
   !> Unlike r2p_film_thickness of the cell alone, it does not read thin where
   !> the cell only clips a thicker film. Huge if there is no film or the plane
   !> misses the stencil. fvf, fbary: film-phase volume fraction and
   !> barycenters of the 3^3 block (gas when the film is gas); lo, hi: the
   !> block's corners; h: the centre cell's sizes.
   real(WP) function r2p_stencil_thickness(fvf,fbary,lo,hi,h,n) result(t)
      implicit none
      real(WP), dimension(-1:1,-1:1,-1:1),   intent(in) :: fvf
      real(WP), dimension(3,-1:1,-1:1,-1:1), intent(in) :: fbary
      real(WP), dimension(3), intent(in) :: lo,hi,h,n
      real(WP), dimension(3) :: moment,ctr
      real(WP) :: volume,cell_vol,area
      integer :: a,b,c
      t=huge(1.0_WP)
      cell_vol=product(h)
      volume=0.0_WP; moment=0.0_WP
      do a=-1,1; do b=-1,1; do c=-1,1
         if (.not.(fvf(a,b,c).gt.0.0_WP)) cycle
         volume=volume+fvf(a,b,c)*cell_vol
         moment=moment+fvf(a,b,c)*cell_vol*fbary(:,a,b,c)
      end do; end do; end do
      if (.not.(volume.gt.0.0_WP)) return
      ctr=moment/volume
      call init_workspace()
      call construct_2pt(ws_block,lo,hi)
      call setNumberOfPlanes(ws_mid,1)
      call setPlane(ws_mid,0,n,dot_product(n,ctr))
      call setFlip(ws_mid,.false.)
      call getPoly(ws_block,ws_mid,0,ws_poly)
      if (getNumberOfVertices(ws_poly).lt.3) return
      area=abs(calculateVolume(ws_poly))
      if (area.le.1.0e-12_WP*cell_vol**(2.0_WP/3.0_WP)) return
      t=volume/area/cell_vol**(1.0_WP/3.0_WP)
   end function r2p_stencil_thickness

   !> Very thin film: PCA slab (C++: r2p_snap.h pcaSlab). Far below the training
   !> range (bag films of ~1e-4 cells) R2P-Net's normals become unreliable and
   !> holes form. Where the film in the 3^3 stencil is at most
   !> r2p_pca_slab_max_thickness cells thick (r2p_stencil_thickness along the
   !> PCA direction), both normals become +-the PCA direction of the 3^3
   !> film-phase barycenters (the network's own input direction; on a thin film
   !> they lie on its mid-surface): a slab. No guard condition: at this
   !> thickness a slab differs negligibly from the real faces even at a film's
   !> end. pca: that direction (physical, unit); other arguments as in
   !> r2p_stencil_thickness. On success n0,n1 are +-pca in the network's
   !> (cell-index) frame, as its own normals are before mesh scaling.
   logical function r2p_pca_slab(fvf,fbary,lo,hi,h,pca,n0,n1) result(slab)
      implicit none
      real(WP), dimension(-1:1,-1:1,-1:1),   intent(in) :: fvf
      real(WP), dimension(3,-1:1,-1:1,-1:1), intent(in) :: fbary
      real(WP), dimension(3), intent(in) :: lo,hi,h,pca
      real(WP), dimension(3), intent(inout) :: n0,n1
      slab=r2p_stencil_thickness(fvf,fbary,lo,hi,h,pca).le.r2p_pca_slab_max_thickness
      if (.not.slab) return
      n0=pca/h; n0=n0/norm2(n0)
      n1=-n0
   end function r2p_pca_slab

   !> Very thin film, for the routing (C++: r2p_snap.h veryThinStencil): the
   !> cells r2p_pca_slab turns into slabs (at least 6 film cells in the 3^3
   !> stencil, r2p_stencil_thickness along their PCA direction at most
   !> r2p_pca_slab_max_thickness). They go to R2P-Net whatever the classifier,
   !> detector or guard says, so they get the slab rather than PLICnet's single
   !> plane. A quick bound skips the PCA for thicker films: the 3^3 block's
   !> largest cross-section is 3^2 sqrt(2) < 16 cell faces, so a film of
   !> thickness t holds less than 16 t cell volumes there. The PCA is that of
   !> the R2P-Net input (vfs pca_normal). Arguments as in r2p_stencil_thickness.
   logical function r2p_very_thin_stencil(fvf,fbary,lo,hi,h,vflo) result(thin)
      implicit none
      real(WP), dimension(-1:1,-1:1,-1:1),   intent(in) :: fvf
      real(WP), dimension(3,-1:1,-1:1,-1:1), intent(in) :: fbary
      real(WP), dimension(3), intent(in) :: lo,hi,h
      real(WP), intent(in) :: vflo
      real(WP), dimension(3,27) :: pts
      real(WP), dimension(3,3) :: cov
      real(WP), dimension(3) :: ctr,dl,ev,pca
      real(WP), dimension(64) :: work
      integer :: a,b,c,np,m,info
      thin=.false.
      if (.not.(r2p_pca_slab_max_thickness.gt.0.0_WP)) return
      if (sum(fvf).gt.16.0_WP*r2p_pca_slab_max_thickness) return
      np=0
      do c=-1,1; do b=-1,1; do a=-1,1
         if (fvf(a,b,c).gt.vflo) then
            np=np+1; pts(:,np)=fbary(:,a,b,c)
         end if
      end do; end do; end do
      if (np.lt.6) return
      ctr=0.0_WP
      do m=1,np; ctr=ctr+pts(:,m); end do
      ctr=ctr/real(np,WP)
      cov=0.0_WP
      do m=1,np
         dl=pts(:,m)-ctr
         cov(1,1)=cov(1,1)+dl(1)*dl(1); cov(1,2)=cov(1,2)+dl(1)*dl(2); cov(1,3)=cov(1,3)+dl(1)*dl(3)
         cov(2,2)=cov(2,2)+dl(2)*dl(2); cov(2,3)=cov(2,3)+dl(2)*dl(3); cov(3,3)=cov(3,3)+dl(3)*dl(3)
      end do
      cov(2,1)=cov(1,2); cov(3,1)=cov(1,3); cov(3,2)=cov(2,3)
      call dsyev('V','U',3,cov,3,ev,work,64,info)
      pca=cov(:,1)/norm2(cov(:,1))
      thin=r2p_stencil_thickness(fvf,fbary,lo,hi,h,pca).le.r2p_pca_slab_max_thickness
   end function r2p_very_thin_stencil

   !> Two planes with exactly opposite normals: a snapped slab (the distance
   !> solve changes only distances, and no network output is exactly
   !> antiparallel). p0,p1: getPlane output (normal, distance).
   logical function r2p_is_slab(p0,p1) result(slab)
      implicit none
      real(WP), dimension(4), intent(in) :: p0,p1
      slab=all(p0(1:3).eq.-p1(1:3))
   end function r2p_is_slab

   !> Thin-film guard slab (C++: r2p_snap.h guardSlab). Where the thin-film guard
   !> holds (r2p_film_guard: the other phase lies on both sides of the film,
   !> apart), the film runs through the cell and needs two planes; but far below
   !> its training range (bag films of ~1e-4 cells) R2P-Net can shrink one normal
   !> below 0.85. The cell then gets a parallel slab around the longer network
   !> normal instead of one plane. False (one plane stays) if both are zero.
   logical function r2p_guard_slab(n0,n1) result(slab)
      implicit none
      real(WP), dimension(3), intent(inout) :: n0,n1
      slab=max(norm2(n0),norm2(n1)).gt.0.0_WP
      if (.not.slab) return
      if (norm2(n0).ge.norm2(n1)) then
         n1=-n0
      else
         n0=-n1
      end if
   end function r2p_guard_slab

   !> Pass 2's fitted normals for a slab cell: rotate the slab, keep it parallel.
   subroutine r2p_keep_slab(n0,n1)
      implicit none
      real(WP), dimension(3), intent(inout) :: n0,n1
      real(WP), dimension(3) :: avg
      avg=n0-n1
      if (norm2(avg).le.0.0_WP) return
      avg=avg/norm2(avg)
      n0=avg; n1=-avg
   end subroutine r2p_keep_slab

   !> Film-tip sensor (C++: r2p_tip_sensor.h filmSpread; the key idea of
   !> detect_lig_edge without the connected-component filter): the spread
   !> (weighted second moment, cell^2) of the film-phase centroids of the 26
   !> neighbours (the cell itself excluded), weighted by the film volume
   !> fraction, about their centre of mass. A film that continues fills a layer
   !> of neighbours (large spread); at a tip it reaches only a few (small). 0 if
   !> the neighbours hold no film. fvf, fbary: film-phase volume fraction and
   !> barycenters of the 3^3 block (gas when the film is gas); xm..dz: cell
   !> centres and sizes of the 3 cells per direction.
   real(WP) function r2p_tip_spread(fvf,fbary,xm,ym,zm,dx,dy,dz,vflo) result(spread)
      implicit none
      real(WP), dimension(-1:1,-1:1,-1:1),   intent(in) :: fvf
      real(WP), dimension(3,-1:1,-1:1,-1:1), intent(in) :: fbary
      real(WP), dimension(-1:1), intent(in) :: xm,ym,zm,dx,dy,dz
      real(WP), intent(in) :: vflo
      real(WP), dimension(3,26) :: p
      real(WP), dimension(26)   :: w
      real(WP), dimension(3)    :: c
      real(WP) :: wsum
      integer :: ii,jj,kk,n,m
      n=0; wsum=0.0_WP; c=0.0_WP
      do ii=-1,1; do jj=-1,1; do kk=-1,1
         if (ii.eq.0.and.jj.eq.0.and.kk.eq.0) cycle
         n=n+1
         w(n)=fvf(ii,jj,kk)
         p(:,n)=[(fbary(1,ii,jj,kk)-xm(ii))/dx(ii)+real(ii,WP),(fbary(2,ii,jj,kk)-ym(jj))/dy(jj)+real(jj,WP), &
         &       (fbary(3,ii,jj,kk)-zm(kk))/dz(kk)+real(kk,WP)]
         wsum=wsum+w(n)
         c=c+w(n)*p(:,n)
      end do; end do; end do
      spread=0.0_WP
      if (wsum.le.vflo) return
      c=c/wsum
      do m=1,n
         spread=spread+w(m)*sum((p(:,m)-c)**2)
      end do
      spread=spread/wsum
   end function r2p_tip_spread

   !> The film's mean normal from R2P-Net's two face normals (grid frame, before
   !> mesh scaling), or the surviving one when the network predicts one plane;
   !> mesh-scaled and normalized (as r2p_edge_sensor.h meanNormal).
   function r2p_mean_normal(n0,n1,dx,dy,dz) result(m)
      implicit none
      real(WP), dimension(3), intent(in) :: n0,n1
      real(WP), intent(in) :: dx,dy,dz
      real(WP), dimension(3) :: m
      real(WP) :: a0,a1
      a0=norm2(n0); a1=norm2(n1)
      if (a0.ge.0.5_WP.and.a1.ge.0.5_WP) then
         m=n0/a0-n1/a1
      else if (a0.ge.a1) then
         m=n0
      else
         m=-n1
      end if
      m=m*[dx,dy,dz]
      m=m/norm2(m)
   end function r2p_mean_normal

   !> Film-edge sensor (C++: r2p_edge_sensor.h edgeCount; NGA2 detect_edge_regions
   !> with the film phase, the network's mean normal and distinct probe cells).
   !> Probes 16 directions, 22.5 deg apart, in the plane normal to `normal`; in
   !> each, the neighbour best aligned with it on the distance-1 ring and on the
   !> distance-2 shell counts as empty if
   !>   presence = sum over its 3^3 cells of min(f/f_centre,1) (f > vflo)
   !> is <= 3 (ring) or <= 0.25 (shell); each probe cell is judged once per
   !> ring. Returns the number of empty probe cells (NGA2: >= 2 = edge).
   !> fvf: film-phase volume fraction over the 7^3 block (index 0 = the cell);
   !> xm, ym, zm: cell centres over -2..2.
   integer function r2p_edge_count(fvf,xm,ym,zm,normal,vflo) result(empty)
      implicit none
      real(WP), dimension(-3:3,-3:3,-3:3), intent(in) :: fvf
      real(WP), dimension(-2:2), intent(in) :: xm,ym,zm
      real(WP), dimension(3), intent(in) :: normal
      real(WP), intent(in) :: vflo
      integer , parameter :: ndir=16
      real(WP), dimension(3) :: axis,t1,t2,d
      real(WP), dimension(3,98) :: dir
      integer , dimension(3,98) :: off
      logical , dimension(98) :: probed
      real(WP) :: th,threshold
      integer :: ring,a,b,c,n,m,q,best,least
      ! Orthonormal tangent basis: cross with the axis least aligned with the normal
      least=1
      if (abs(normal(2)).lt.abs(normal(least))) least=2
      if (abs(normal(3)).lt.abs(normal(least))) least=3
      axis=0.0_WP; axis(least)=1.0_WP
      t1=cross(normal,axis); t1=t1/norm2(t1)
      t2=cross(normal,t1)
      empty=0
      do ring=1,2
         ! Unit direction to every cell of the ring (Chebyshev distance `ring`)
         n=0
         do a=-ring,ring; do b=-ring,ring; do c=-ring,ring
            if (max(abs(a),abs(b),abs(c)).ne.ring) cycle
            n=n+1
            off(:,n)=[a,b,c]
            dir(:,n)=[xm(a)-xm(0),ym(b)-ym(0),zm(c)-zm(0)]
            dir(:,n)=dir(:,n)/norm2(dir(:,n))
         end do; end do; end do
         ! Best-aligned cell for each direction; each distinct cell judged once
         probed=.false.
         threshold=merge(3.0_WP,0.25_WP,ring.eq.1)
         do m=0,ndir-1
            th=2.0_WP*acos(-1.0_WP)*real(m,WP)/real(ndir,WP)
            d=cos(th)*t1+sin(th)*t2
            best=1
            do q=2,n
               if (dot_product(d,dir(:,q)).gt.dot_product(d,dir(:,best))) best=q
            end do
            if (probed(best)) cycle
            probed(best)=.true.
            if (presence(off(1,best),off(2,best),off(3,best)).le.threshold) empty=empty+1
         end do
      end do
   contains
      function cross(u,v) result(w)
         real(WP), dimension(3), intent(in) :: u,v
         real(WP), dimension(3) :: w
         w=[u(2)*v(3)-u(3)*v(2),u(3)*v(1)-u(1)*v(3),u(1)*v(2)-u(2)*v(1)]
      end function cross
      !> Film near cell (i0,j0,k0), relative to the centre cell
      real(WP) function presence(i0,j0,k0) result(s)
         integer, intent(in) :: i0,j0,k0
         integer :: ii,jj,kk
         s=0.0_WP
         do ii=i0-1,i0+1; do jj=j0-1,j0+1; do kk=k0-1,k0+1
            if (fvf(ii,jj,kk).gt.vflo) s=s+min(fvf(ii,jj,kk)/fvf(0,0,0),1.0_WP)
         end do; end do; end do
      end function presence
   end function r2p_edge_count

   !> Topological film-edge sensor (C++: r2p_edge_topology.h gasWraps). Above
   !> and below a film, the other phase forms two regions that meet only where
   !> the film ends. Every cell a film passes through holds some film, and the
   !> cells cut by a surface separate the cells on its two sides when those are
   !> joined through faces only, so a continuing film (however thin, tilted or
   !> curved) keeps the two regions apart, and so does a thick rim. Seeds one
   !> empty cell on each side of the film (walking from the cell along +-normal
   !> up to 2 cells), floods the empty cells of the 5^3 block through faces
   !> from one seed, and returns 1 (edge) if the flood reaches the other seed:
   !> the film ends within about 2 cells; 0 if not; 2 if a side has no empty
   !> cell within 2 cells and the farthest cell walked there holds film VF >=
   !> thick_film (a film thicker than that, e.g. a thick rounded rim: no thin
   !> film to keep from pinching off). A thin film cannot reach that far cell,
   !> so a side blocked only by stray traces of film in the other phase gives 0:
   !> the film is thin.
   !> fvf: film-phase volume fraction over the 5^3 block (index 0 = the cell),
   !> empty where <= vflo; h: the cell's sizes; normal: the film's mean normal.
   integer function r2p_edge_topology(fvf,h,normal,vflo) result(edge)
      implicit none
      integer , parameter :: reach=2
      real(WP), parameter :: thick_film=0.01_WP  !< film VF in the farthest walked cell of a blocked side
      real(WP), dimension(-reach:reach,-reach:reach,-reach:reach), intent(in) :: fvf
      real(WP), dimension(3), intent(in) :: h,normal
      real(WP), intent(in) :: vflo
      logical , dimension(-reach:reach,-reach:reach,-reach:reach) :: empty,reached
      integer , dimension(3,(2*reach+1)**3) :: stack
      integer , dimension(3,2) :: seed
      logical , dimension(2) :: found
      integer , dimension(3) :: c,q,n
      real(WP), dimension(3) :: d
      real(WP) :: dmax,sgn,far
      logical :: thick
      integer :: side,step,a,top,f
      edge=0
      empty=fvf.le.vflo
      ! Seeds: the first empty cell walking from the cell along +normal and
      ! along -normal, in Chebyshev steps of one cell
      dmax=0.0_WP
      do a=1,3
         d(a)=normal(a)/h(a)
         dmax=max(dmax,abs(d(a)))
      end do
      if (.not.dmax.gt.0.0_WP) return
      found=.false.
      thick=.false.
      do side=1,2
         sgn=merge(1.0_WP,-1.0_WP,side.eq.1)
         far=0.0_WP   ! film in the farthest cell walked on this side
         do step=1,reach
            do a=1,3
               c(a)=nint(sgn*real(step,WP)*d(a)/dmax)
            end do
            if (empty(c(1),c(2),c(3))) then
               seed(:,side)=c; found(side)=.true.; exit
            end if
            far=fvf(c(1),c(2),c(3))
         end do
         if (.not.found(side).and.far.ge.thick_film) thick=.true.
      end do
      if (thick) then
         edge=2; return
      end if
      if (.not.all(found)) return
      ! Flood the empty cells through faces from the first seed
      reached=.false.
      top=1; stack(:,1)=seed(:,1); reached(seed(1,1),seed(2,1),seed(3,1))=.true.
      do while (top.gt.0)
         q=stack(:,top); top=top-1
         if (all(q.eq.seed(:,2))) then
            edge=1; return
         end if
         do f=1,6
            n=q; a=(f+1)/2; n(a)=n(a)+merge(-1,1,mod(f,2).eq.1)
            if (any(abs(n).gt.reach)) cycle
            if (.not.empty(n(1),n(2),n(3)).or.reached(n(1),n(2),n(3))) cycle
            reached(n(1),n(2),n(3))=.true.
            top=top+1; stack(:,top)=n
         end do
      end do
   end function r2p_edge_topology

   !> Thin-film guard for the R2P routing (C++: r2p_edge_topology.h
   !> filmSeparates): true where the other phase lies on both sides of the film
   !> around the cell, in separate regions, so the film continues through the
   !> cell and needs two planes, whatever the classifier or detector says. The
   !> empty cells of the 5^3 block (film-phase VF <= vflo) are split into
   !> regions joined through faces; true if at least two regions each reach
   !> both the cell's 3^3 neighbourhood (the other phase within about a cell of
   !> it on that side) and the edge of the block (not an enclosed pocket). A
   !> droplet, a ligament or a film's end leaves the other phase in one region
   !> around it; a resolved interface has it on one side only. Films up to
   !> about a cell thick qualify (a diagonal one up to ~0.9 cells). Where the
   !> film in the 3^3 neighbourhood is thin (volume <= thin_volume), "near"
   !> reaches to squared distance near_dist2: a thin film diagonal to the grid
   !> clips the corners of a third row of cells, which can fill the 3^3
   !> neighbourhood on one side and leave the other phase there only at the
   !> diagonal (0,1,2) neighbour.
   !> fvf: film-phase volume fraction over the 5^3 block (index 0 = the cell).
   logical function r2p_film_guard(fvf,vflo) result(film)
      implicit none
      integer , parameter :: reach=2
      real(WP), parameter :: thin_volume=1.0_WP   !< film volume in the 3^3 neighbourhood, cells
      integer , parameter :: near_dist2=5          !< squared distance, cells
      real(WP), dimension(-reach:reach,-reach:reach,-reach:reach), intent(in) :: fvf
      real(WP), intent(in) :: vflo
      logical , dimension(-reach:reach,-reach:reach,-reach:reach) :: empty,done
      integer , dimension(3,(2*reach+1)**3) :: stack
      integer , dimension(3) :: q,n
      integer :: a,b,c,top,f,ax,regions,dist2
      logical :: near,outer
      film=.false.
      empty=fvf.le.vflo
      dist2=3   ! exactly the 3^3 neighbourhood
      if (sum(fvf(-1:1,-1:1,-1:1)).le.thin_volume) dist2=near_dist2
      done=.false.
      regions=0
      do a=-reach,reach; do b=-reach,reach; do c=-reach,reach
         if (.not.empty(a,b,c).or.done(a,b,c)) cycle
         ! Flood one region through faces; does it come near the cell and reach the block's edge?
         near=.false.; outer=.false.
         top=1; stack(:,1)=[a,b,c]; done(a,b,c)=.true.
         do while (top.gt.0)
            q=stack(:,top); top=top-1
            if (sum(q**2).le.dist2) near=.true.
            if (maxval(abs(q)).eq.reach) outer=.true.
            do f=1,6
               n=q; ax=(f+1)/2; n(ax)=n(ax)+merge(-1,1,mod(f,2).eq.1)
               if (any(abs(n).gt.reach)) cycle
               if (.not.empty(n(1),n(2),n(3)).or.done(n(1),n(2),n(3))) cycle
               done(n(1),n(2),n(3))=.true.
               top=top+1; stack(:,top)=n
            end do
         end do
         if (near.and.outer) regions=regions+1
         if (regions.ge.2) then
            film=.true.; return
         end if
      end do; end do; end do
   end function r2p_film_guard

   !> Whether pinch prevention treats the cell as an edge (the planes may meet):
   !> the topological sensor's verdict if r2p_edge_use_topology, else the
   !> presence sensor's (C++: r2p_nopinch.h isEdge). The topological sensor is
   !> overruled where the thin-film guard holds (guard /= 0): the guard found
   !> the other phase on the two sides of the film in separate regions without
   !> using a normal, so the film continues; the sensor's seeds, walked along
   !> R2P-Net's mean normal, can land on one side of a very thin film whose
   !> normals are poor and call it an edge. A thick film (edge_topo 2: no empty
   !> cell within 2 cells on a side) is skipped too: one cell's planes cannot
   !> pinch it off, and at a thick rounded rim opening the converging planes
   !> would push liquid past the real end.
   logical function r2p_is_edge(edge_presence,edge_topo,guard) result(is_edge)
      implicit none
      real(WP), intent(in) :: edge_presence,edge_topo,guard
      if (r2p_edge_use_topology) then
         is_edge=edge_topo.ge.2.0_WP.or.(edge_topo.ge.1.0_WP.and.guard.eq.0.0_WP)
      else
         is_edge=edge_presence.ge.r2p_edge_min_count
      end if
   end function r2p_is_edge

   !> Pinch prevention (C++: r2p_nopinch.h). Two Newton-placed planes can meet
   !> inside the cell, giving the film zero thickness there: at a real edge
   !> that is the tip (the caller skips edges), elsewhere it pinches the film
   !> off. If the film is anywhere in the cell thinner than
   !> r2p_nopinch_gap_fraction times its mean thickness (along
   !> the mean normal m; the gap is linear, so its minimum is at a vertex of
   !> cell ∩ film, found exactly), both normals are rotated toward m, each
   !> keeping its share of the opening,
   !>   n_i(l) = normalize((n_i.m) m + (1-l)(n_i - (n_i.m) m)),
   !> with the distances re-solved, taking the smallest l in [0,1] (bisection)
   !> that restores the bound (l = 1: a parallel slab). Returns l (0: unchanged).
   real(WP) function r2p_prevent_pinch_planar(cell,sep,vf,lbary,gbary) result(l)
      implicit none
      type(RectCub_type),   intent(inout) :: cell
      type(PlanarSep_type), intent(inout) :: sep
      real(WP), intent(in) :: vf
      real(WP), dimension(3), intent(in) :: lbary,gbary
      real(WP), dimension(4) :: plane
      real(WP), dimension(3) :: lo,hi,n0,n1,m,cen
      real(WP) :: l_lo,l_hi,mid,s,t_ref,width,film_vf,area
      logical :: flipped
      integer :: it
      l=0.0_WP
      if (r2p_nopinch_gap_fraction.lt.0.0_WP.or.getNumberOfPlanes(sep).ne.2) return
      call init_workspace()
      call getBoundingPts(cell,lo,hi)
      flipped=isFlipped(sep)
      cen=lbary; if (flipped) cen=gbary
      plane=getPlane(sep,0); n0=plane(1:3)
      plane=getPlane(sep,1); n1=plane(1:3)
      s=1.0_WP; if (flipped) s=-1.0_WP
      m=s*n0-s*n1; m=m/norm2(m)
      ! Mean film thickness: film volume over the area of the plane through the
      ! film centroid along m, clipped to the cell (a full cross-section if
      ! that plane misses the cell). Always positive, so planes that meet in
      ! the cell (gap 0, or ~1e-17 from rounding) never pass; the gap at the
      ! film centroid, used before, is <= 0 when the centroid lies beyond where
      ! the planes meet, which let such planes through.
      width=hi(1)-lo(1)
      film_vf=vf; if (flipped) film_vf=1.0_WP-vf
      t_ref=film_vf*width
      call setNumberOfPlanes(ws_mid,1)
      call setPlane(ws_mid,0,m,dot_product(m,cen))
      call setFlip(ws_mid,.false.)
      call getPoly(cell,ws_mid,0,ws_poly)
      if (getNumberOfVertices(ws_poly).ge.3) then
         area=abs(calculateVolume(ws_poly))
         if (area.gt.1.0e-12_WP*product(hi-lo)**(2.0_WP/3.0_WP)) t_ref=film_vf*product(hi-lo)/area
      end if
      if (thick_enough(sep)) return
      l_lo=0.0_WP; l_hi=1.0_WP
      do it=1,14
         mid=0.5_WP*(l_lo+l_hi)
         call solve(mid)
         if (thick_enough(ws_np)) then
            l_hi=mid
         else
            l_lo=mid
         end if
      end do
      call solve(l_hi)
      call copy(sep,ws_np)
      l=l_hi
   contains
      !> ws_np = the two planes rotated by lam, Newton-placed
      subroutine solve(lam)
         real(WP), intent(in) :: lam
         call setNumberOfPlanes(ws_np,2)
         call setPlane(ws_np,0,rotated(n0,lam),0.0_WP)
         call setPlane(ws_np,1,rotated(n1,lam),0.0_WP)
         call setFlip(ws_np,flipped)
         call r2p_newton_distances_planar(cell,ws_np,vf,lbary,gbary)
      end subroutine solve
      function rotated(n,lam) result(r)
         real(WP), dimension(3), intent(in) :: n
         real(WP), intent(in) :: lam
         real(WP), dimension(3) :: r,along
         along=dot_product(n,m)*m
         r=along+(1.0_WP-lam)*(n-along)
         r=r/norm2(r)
      end function rotated
      !> Film no thinner than the bound anywhere in the cell (true with < 2 planes)
      logical function thick_enough(t) result(ok)
         type(PlanarSep_type), intent(inout) :: t
         real(WP), dimension(3) :: nf(3,2),mf
         real(WP), dimension(2) :: df
         real(WP), dimension(4) :: q
         real(WP) :: sf,g
         integer :: p
         ok=.true.
         if (getNumberOfPlanes(t).ne.2) return
         sf=1.0_WP; if (isFlipped(t)) sf=-1.0_WP
         do p=1,2
            q=getPlane(t,p-1); nf(:,p)=sf*q(1:3); df(p)=sf*q(4)
         end do
         mf=nf(:,1)-nf(:,2); mf=mf/norm2(mf)
         g=film_min_gap(nf,df,mf,lo,hi)
         ok=(g.ge.r2p_nopinch_gap_fraction*t_ref.and.g.gt.1.0e-12_WP*width)
      end function thick_enough
   end function r2p_prevent_pinch_planar

   !> r2p_prevent_pinch on a SeparatorVariant (NGA2's liquid_gas_interface):
   !> planes and flip copied to a PlanarSep, treated there, and copied back
   real(WP) function r2p_prevent_pinch_variant(cell,sep,vf,lbary,gbary) result(l)
      implicit none
      type(RectCub_type),          intent(inout) :: cell
      type(SeparatorVariant_type), intent(inout) :: sep
      real(WP), intent(in) :: vf
      real(WP), dimension(3), intent(in) :: lbary,gbary
      real(WP), dimension(4) :: plane
      integer :: p
      l=0.0_WP
      if (r2p_nopinch_gap_fraction.lt.0.0_WP.or.getNumberOfPlanes(sep).ne.2) return
      call init_workspace()
      call setNumberOfPlanes(ws_sep,2)
      do p=0,1
         plane=getPlane(sep,p)
         call setPlane(ws_sep,p,plane(1:3),plane(4))
      end do
      call setFlip(ws_sep,logical(isFlipped(sep)))
      l=r2p_prevent_pinch_planar(cell,ws_sep,vf,lbary,gbary)
      if (l.le.0.0_WP) return
      call setNumberOfPlanes(sep,getNumberOfPlanes(ws_sep))
      do p=0,getNumberOfPlanes(ws_sep)-1
         plane=getPlane(ws_sep,p)
         call setPlane(sep,p,plane(1:3),plane(4))
      end do
      call setFlip(sep,logical(isFlipped(ws_sep)))
   end function r2p_prevent_pinch_variant

   !> The planes of sep as [n0,d0,n1,d1] (zeros past its number of planes)
   function r2p_planes(sep) result(p)
      implicit none
      type(SeparatorVariant_type), intent(inout) :: sep
      real(WP), dimension(8) :: p
      real(WP), dimension(4) :: q
      integer :: n
      p=0.0_WP
      do n=0,min(getNumberOfPlanes(sep),2)-1
         q=getPlane(sep,n)
         p(4*n+1:4*n+4)=q
      end do
   end function r2p_planes

   !> Thinnest film of the two planes of sep inside the cell, measured along
   !> their mean normal: <= 0 where they cross or meet in the cell; -huge with
   !> a NaN or a degenerate mean normal; huge with fewer than two planes
   real(WP) function r2p_sep_min_gap(cell,sep) result(g)
      use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
      implicit none
      type(RectCub_type),          intent(inout) :: cell
      type(SeparatorVariant_type), intent(inout) :: sep
      real(WP), dimension(3,2) :: nf
      real(WP), dimension(2) :: df
      real(WP), dimension(3) :: mf,lo,hi
      real(WP), dimension(4) :: q
      real(WP) :: sf
      integer :: p
      g=huge(1.0_WP)
      if (getNumberOfPlanes(sep).ne.2) return
      g=-huge(1.0_WP)
      sf=1.0_WP; if (isFlipped(sep)) sf=-1.0_WP
      do p=1,2
         q=getPlane(sep,p-1)
         if (any(ieee_is_nan(q))) return
         nf(:,p)=sf*q(1:3); df(p)=sf*q(4)
      end do
      mf=nf(:,1)-nf(:,2)
      if (.not.norm2(mf).gt.0.0_WP) return
      mf=mf/norm2(mf)
      call getBoundingPts(cell,lo,hi)
      g=film_min_gap(nf,df,mf,lo,hi)
   end function r2p_sep_min_gap

   !> Do the two planes of sep cross or meet inside the cell (or hold a NaN)?
   !> Thinnest film at most 1e-12 cell widths: where planes meet, the gap
   !> computed at the meeting line is 0 only up to rounding (~1e-17)
   logical function r2p_planes_cross(cell,sep) result(crossed)
      implicit none
      type(RectCub_type),          intent(inout) :: cell
      type(SeparatorVariant_type), intent(inout) :: sep
      real(WP), dimension(3) :: lo,hi
      crossed=.false.
      if (getNumberOfPlanes(sep).ne.2) return
      call getBoundingPts(cell,lo,hi)
      crossed=.not.(r2p_sep_min_gap(cell,sep).gt.1.0e-12_WP*(hi(1)-lo(1)))
   end function r2p_planes_cross

   !> Self-check report of one cell, for offline replay (r2p_nopinch_debug):
   !> stage, indices and flip; cell box; VF and barycenters; presence and
   !> topological edge sensors and unpinch; the network normals (grid frame,
   !> before mesh scaling); the planes [n0,d0,n1,d1] after the pass-1 Newton
   !> solve and after its pinch prevention; the pass-2 fitted normals and its
   !> planes after the Newton solve (zeros outside pass 2); the current planes
   !> and their thinnest film in the cell.
   !> dbg: network normals (1:6), pass-1 planes after Newton (7:14) and after
   !> pinch prevention (15:22)
   subroutine r2p_nopinch_report(stage,i,j,k,cell,sep,vf,lbary,gbary,edge,edge_topo,unpinch,dbg,fit,pass2_newton)
      implicit none
      character(len=*), intent(in) :: stage
      integer, intent(in) :: i,j,k
      type(RectCub_type),          intent(inout) :: cell
      type(SeparatorVariant_type), intent(inout) :: sep
      real(WP), intent(in) :: vf,edge,edge_topo,unpinch
      real(WP), dimension(3),  intent(in) :: lbary,gbary
      real(WP), dimension(22), intent(in) :: dbg
      real(WP), dimension(6),  intent(in) :: fit
      real(WP), dimension(8),  intent(in) :: pass2_newton
      character(len=*), parameter :: tag='R2P_NOPINCH_DEBUG',fmt='(a,1x,a,*(1x,es24.16))'
      real(WP), dimension(3) :: lo,hi
      if (r2p_nopinch_reports.ge.r2p_nopinch_debug_max) return
      r2p_nopinch_reports=r2p_nopinch_reports+1
      call getBoundingPts(cell,lo,hi)
      write(*,'(a,1x,a,3(1x,i0),a,l1)') tag,stage,i,j,k,' flipped=',logical(isFlipped(sep))
      write(*,fmt) tag,'box',lo,hi
      write(*,fmt) tag,'vf_lbary_gbary',vf,lbary,gbary
      write(*,fmt) tag,'edge_edgetopo_unpinch',edge,edge_topo,unpinch
      write(*,fmt) tag,'network_normals',dbg(1:6)
      write(*,fmt) tag,'pass1_newton',dbg(7:14)
      write(*,fmt) tag,'pass1_unpinched',dbg(15:22)
      write(*,fmt) tag,'pass2_fit',fit
      write(*,fmt) tag,'pass2_newton',pass2_newton
      write(*,fmt) tag,'current',r2p_planes(sep)
      write(*,fmt) tag,'current_min_gap',r2p_sep_min_gap(cell,sep)
   end subroutine r2p_nopinch_report

   !> Film thickness along m through x, for the film n_p.x <= d_p (negative
   !> where the planes have crossed)
   real(WP) function film_gap_at(n,d,m,x) result(g)
      implicit none
      real(WP), dimension(3,2), intent(in) :: n
      real(WP), dimension(2), intent(in) :: d
      real(WP), dimension(3), intent(in) :: m,x
      real(WP) :: top,bot,nm,t
      integer :: p
      top=huge(1.0_WP); bot=-huge(1.0_WP)
      do p=1,2
         nm=dot_product(n(:,p),m)
         if (abs(nm).lt.1.0e-12_WP) cycle
         t=(d(p)-dot_product(n(:,p),x))/nm
         if (nm.gt.0.0_WP) then
            top=min(top,t)
         else
            bot=max(bot,t)
         end if
      end do
      g=top-bot
   end function film_gap_at

   !> Thinnest film inside the box [lo,hi]: the gap at the vertices of box ∩
   !> film (0 on the line where the planes meet); huge if the film misses it
   real(WP) function film_min_gap(n,d,m,lo,hi) result(g)
      implicit none
      real(WP), dimension(3,2), intent(in) :: n
      real(WP), dimension(2), intent(in) :: d
      real(WP), dimension(3), intent(in) :: m,lo,hi
      real(WP), dimension(3,8) :: A
      real(WP), dimension(8) :: B
      real(WP), dimension(3) :: bc,x
      real(WP) :: det,tol
      integer :: a1,b1,c1,k,q
      logical :: inside
      A=0.0_WP
      do k=1,3
         A(k,2*k-1)=1.0_WP;  B(2*k-1)=hi(k)
         A(k,2*k)=-1.0_WP;   B(2*k)=-lo(k)
      end do
      A(:,7)=n(:,1); B(7)=d(1)
      A(:,8)=n(:,2); B(8)=d(2)
      tol=1.0e-10_WP*(hi(1)-lo(1))
      g=huge(1.0_WP)
      do a1=1,8; do b1=a1+1,8; do c1=b1+1,8
         bc=cross3(A(:,b1),A(:,c1))
         det=dot_product(A(:,a1),bc)
         if (abs(det).lt.1.0e-12_WP) cycle
         x=(B(a1)*bc+B(b1)*cross3(A(:,c1),A(:,a1))+B(c1)*cross3(A(:,a1),A(:,b1)))/det
         inside=.true.
         do q=1,8
            if (dot_product(A(:,q),x).gt.B(q)+tol) then
               inside=.false.; exit
            end if
         end do
         if (inside) g=min(g,film_gap_at(n,d,m,x))
      end do; end do; end do
   contains
      function cross3(u,v) result(w)
         real(WP), dimension(3), intent(in) :: u,v
         real(WP), dimension(3) :: w
         w=[u(2)*v(3)-u(3)*v(2),u(3)*v(1)-u(1)*v(3),u(1)*v(2)-u(2)*v(1)]
      end function cross3
   end function film_min_gap

end module r2p_net_tools
