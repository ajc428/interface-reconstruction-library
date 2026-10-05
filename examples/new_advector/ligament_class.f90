!> Definition for a ligament atomization class
module ligament_class
   use string,            only: str_medium
   use precision,         only: WP
   use inputfile_class,   only: inputfile
   use config_class,      only: config
   use iterator_class,    only: iterator
   use ensight_class,     only: ensight
   use surfmesh_class,    only: surfmesh
   use partmesh_class,    only: partmesh
   use hypre_str_class,   only: hypre_str
   !use ddadi_class,       only: ddadi
   use vfs_class,         only: vfs
   use tpns_class,        only: tpns
   use timetracker_class, only: timetracker
   use event_class,       only: event
   use pardata_class,     only: pardata
   use monitor_class,     only: monitor
   use lpt_class,         only: lpt
   use cclabel_class,     only: cclabel
   use irl_fortran_interface
   implicit none
   private
   
   public :: ligament
   
   !> Ligament object
   type :: ligament
      
      !> Provide a pardata and an event tracker for saving restarts
      type(event)    :: save_evt
      type(pardata)  :: df
      character(len=str_medium) :: lpt_file
      logical :: restarted

      !> Input file for the simulation
      type(inputfile) :: input

      !> Config
      type(config) :: cfg
      
      !> Flow solver
      type(vfs)         :: vf    !< Volume fraction solver
      type(tpns)        :: fs    !< Two-phase flow solver
      type(hypre_str)   :: ps    !< Structured Hypre linear solver for pressure
      !type(ddadi)       :: vs    !< DDADI solver for velocity
      type(timetracker) :: time  !< Time info
      type(cclabel)     :: ccl,ccl_film,ccl_lig
      !type(transfermodels) :: tm
      
      !> Ensight postprocessing
      type(surfmesh) :: smesh    !< Surface mesh for interface
      type(ensight)  :: ens_out  !< Ensight output for flow variables
      type(event)    :: ens_evt  !< Event trigger for Ensight output
      
      !> Simulation monitor file
      type(monitor) :: mfile    !< General simulation monitoring
      type(monitor) :: cflfile  !< CFL monitoring
      type(monitor) :: dropfile !< Droplet statistics monitoring
      
      !> Work arrays
      real(WP), dimension(:,:,:), allocatable :: resU,resV,resW      !< Residuals
      real(WP), dimension(:,:,:), allocatable :: Ui,Vi,Wi            !< Cell-centered velocities

      !> Iterator for VOF removal
      type(iterator) :: vof_removal_layer  !< Edge of domain where we actively remove VOF
      real(WP) :: vof_removed              !< Integral of VOF removed

      !> Drop transfer modeling
      logical :: use_drop_transfer !< Do we use droplet transfer
      logical :: use_film_transfer !< Do we use film transfer
      logical :: use_lig_transfer  !< Do we use ligament transfer
      logical :: use_secondary     !< Do we use secondary breakup
      type(lpt)      :: lp         !< Lagrangian particle tracking
      type(monitor)  :: pfile      !< Particle monitoring
      type(partmesh) :: pmesh      !< Particle mesh for lpt
      
      
   contains
      procedure :: init                            !< Initialize nozzle simulation
      procedure :: step                            !< Advance nozzle simulation by one time step
      procedure :: final                           !< Finalize nozzle simulation
   end type ligament
   

contains
   
   !> Function that defines a level set function for a droplet
   function levelset_droplet(xyz,t) result(G)
      implicit none
      real(WP), dimension(3),intent(in) :: xyz
      real(WP), intent(in) :: t
      real(WP) :: G
      G=0.5_WP-sqrt(xyz(1)**2+xyz(2)**2+xyz(3)**2)
   end function levelset_droplet


   !> Function that defines a level set function for a ligament
   function levelset_ligament(xyz,t) result(G)
      implicit none
      real(WP), dimension(3),intent(in) :: xyz
      real(WP), intent(in) :: t
      real(WP) :: G
      G=0.5_WP-sqrt(xyz(1)**2+xyz(2)**2)
   end function levelset_ligament

   !> Initialization of ligament simulation
   subroutine init(this)
      implicit none
      class(ligament), intent(inout) :: this
      
      ! Setup an input file
      read_input: block
         use parallel, only: amRoot
         this%input=inputfile(amRoot=amRoot,filename='input')
      end block read_input

      ! Create the ligament mesh
      create_config: block
         use sgrid_class, only: cartesian,sgrid
         use param,       only: param_read
         use parallel,    only: group
         real(WP), dimension(:), allocatable :: x,y,z
         integer, dimension(3) :: partition
         type(sgrid) :: grid
         integer :: i,j,k,nx,ny,nz
         real(WP) :: Lx,Ly,Lz,xlig
         ! Read in grid definition
         call param_read('Lx',Lx); call param_read('nx',nx); allocate(x(nx+1)); call param_read('X ligament',xlig)
         call param_read('Ly',Ly); call param_read('ny',ny); allocate(y(ny+1))
         call param_read('Lz',Lz); call param_read('nz',nz); allocate(z(nz+1))
         ! Create simple rectilinear grid
         do i=1,nx+1
            x(i)=real(i-1,WP)/real(nx,WP)*Lx-xlig
         end do
         do j=1,ny+1
            y(j)=real(j-1,WP)/real(ny,WP)*Ly-0.5_WP*Ly
         end do
         do k=1,nz+1
            z(k)=real(k-1,WP)/real(nz,WP)*Lz-0.5_WP*Lz
         end do
         ! General serial grid object
         grid=sgrid(coord=cartesian,no=3,x=x,y=y,z=z,xper=.false.,yper=.true.,zper=.true.,name='Ligament')
         ! Read in partition
         call param_read('Partition',partition,short='p')
         ! Create partitioned grid without walls
         this%cfg=config(grp=group,decomp=partition,grid=grid)
      end block create_config
      

      ! Initialize time tracker with 2 subiterations
      initialize_timetracker: block
         use param, only: param_read
         this%time=timetracker(amRoot=this%cfg%amRoot)
         call param_read('Max timestep size',this%time%dtmax)
         call param_read('Max cfl number',this%time%cflmax)
         call param_read('Max time',this%time%tmax)
         this%time%dt=this%time%dtmax
         this%time%itmax=2
      end block initialize_timetracker

      ! Handle restart/saves here
      restart_and_save: block
         use param,                 only: param_read
         use string,                only: str_medium
         use filesys,               only: makedir,isdir
         use irl_fortran_interface
         character(len=str_medium) :: timestamp
         integer, dimension(3) :: iopartition

         ! Create event for saving restart files
         this%save_evt=event(this%time,'Restart output')
         call this%input%read('Restart output period',this%save_evt%tper)
         ! Check if we are restarting
         call this%input%read('Restart from',timestamp,default='')
         this%restarted=.false.; if (len_trim(timestamp).gt.0) this%restarted=.true.
         ! Read in the I/O partition
         call this%input%read('I/O partition',iopartition)
         ! Perform pardata initialization
         if (this%restarted) then
            ! We are restarting, read the file
            call this%df%initialize(pg=this%cfg,iopartition=iopartition,fdata='restart/data_'//trim(adjustl(timestamp)))
         else
            ! We are not restarting, prepare a new directory for storing restart files
            if (this%cfg%amRoot) then
               if (.not.isdir('restart')) call makedir('restart')
            end if
            ! Prepare pardata object for saving restart files
            call this%df%initialize(pg=this%cfg,iopartition=iopartition,filename=trim(this%cfg%name),nval=2,nvar=24)
            this%df%valname=['t ','dt']
            this%df%varname=['U  ','V  ','W  ','P  ','Pjx','Pjy','Pjz','P11','P12','P13','P14','P21','P22','P23','P24','Cr ','Cf ','Co1','Co2','Co3','C11','C12','C13','RT ']
         end if
      end block restart_and_save
      
      
      ! Allocate work arrays
      allocate_work_arrays: block
         allocate(this%resU(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_))
         allocate(this%resV(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_))
         allocate(this%resW(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_))
         allocate(this%Ui  (this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_))
         allocate(this%Vi  (this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_))
         allocate(this%Wi  (this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_))
      end block allocate_work_arrays
      
      
      ! Initialize our VOF solver and field
      create_and_initialize_vof: block
         use vfs_class, only: VFlo,VFhi,elvira,r2p,lvira,plicnet,remap,plic_cylinder,r2pnet,r2p_cylinder,r2p_net
         use mms_geom,  only: cube_refine_vol
         use param,     only: param_read
         integer :: i,j,k,n,si,sj,sk
         real(WP), dimension(3,8) :: cube_vertex
         real(WP), dimension(3) :: v_cent,a_cent
         real(WP) :: vol,area
         integer, parameter :: amr_ref_lvl=4
         real(WP), dimension(:,:,:), allocatable :: P11,P12,P13,P14
         real(WP), dimension(:,:,:), allocatable :: P21,P22,P23,P24
         real(WP), dimension(:,:,:), allocatable :: Cr,Co1,Co2,Co3,Cn11,Cn12,Cn13,RT,Cf
         real(WP), dimension(3) :: datum
         real(WP), dimension(3) :: ref1,ref2,ref3
         logical :: partfile_exists
         real(WP) :: n1, n2, n3
         real(WP) :: mag
         real(WP), dimension(3) :: v1
         ! Create a VOF solver
         !call this%vf%initialize(cfg=this%cfg,reconstruction_method=r2p,name='VOF')
	      !this%vf=vfs(cfg=this%cfg,reconstruction_method=lvira,name='VOF')
	      !this%vf=vfs(cfg=this%cfg,reconstruction_method=elvira,name='VOF')
	      !this%vf=vfs(cfg=this%cfg,reconstruction_method=ml,name='VOF')
		   !this%vf=vfs(cfg=this%cfg,reconstruction_method=ml2,name='VOF')
         !call this%vf%initialize(cfg=this%cfg,reconstruction_method=plicnet,transport_method=remap,name='VOF')
         call this%vf%initialize(cfg=this%cfg,reconstruction_method=r2p_net,transport_method=remap,name='VOF')
         ! Perform pardata initialization
         if (this%restarted) then
            ! We are restarting, read the file
            allocate(P11(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P11',var=P11)
            allocate(P12(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P12',var=P12)
            allocate(P13(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P13',var=P13)
            allocate(P14(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P14',var=P14)
            allocate(P21(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P21',var=P21)
            allocate(P22(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P22',var=P22)
            allocate(P23(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P23',var=P23)
            allocate(P24(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='P24',var=P24)
            allocate(Cr(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='Cr',var=Cr)
            allocate(Co1(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='Co1',var=Co1)
            allocate(Co2(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='Co2',var=Co2)
            allocate(Co3(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='Co3',var=Co3)
            allocate(Cn11(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='C11',var=Cn11)
            allocate(Cn12(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='C12',var=Cn12)
            allocate(Cn13(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='C13',var=Cn13)
            allocate(RT(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='RT',var=RT)
            allocate(Cf(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); call this%df%pull(name='Cf',var=Cf)  
            do k=this%vf%cfg%kmino_,this%vf%cfg%kmaxo_
               do j=this%vf%cfg%jmino_,this%vf%cfg%jmaxo_
                  do i=this%vf%cfg%imino_,this%vf%cfg%imaxo_
                     if (i.lt.this%vf%cfg%imin.or.i.gt.this%vf%cfg%imax.or.j.lt.this%vf%cfg%jmin.or.j.gt.this%vf%cfg%jmax.or.k.lt.this%vf%cfg%kmin.or.k.gt.this%vf%cfg%kmax) then
                        P11(i,j,k) = 0.0_WP
                        P12(i,j,k) = 0.0_WP
                        P13(i,j,k) = 0.0_WP
                        P14(i,j,k) = 0.0_WP
                        P21(i,j,k) = 0.0_WP
                        P22(i,j,k) = 0.0_WP
                        P23(i,j,k) = 0.0_WP
                        P24(i,j,k) = 0.0_WP
                        Cr(i,j,k)  = 0.0_WP
                        Cf(i,j,k)  = 0.0_WP
                        Co1(i,j,k) = 0.0_WP
                        Co2(i,j,k) = 0.0_WP
                        Co3(i,j,k) = 0.0_WP
                        Cn11(i,j,k)= 0.0_WP
                        Cn12(i,j,k)= 0.0_WP
                        Cn13(i,j,k)= 0.0_WP
                        RT(i,j,k)  = 0.0_WP
                     end if
                     ! Check if the second plane is meaningful
                     if (this%vf%two_planes.and.P21(i,j,k)**2+P22(i,j,k)**2+P23(i,j,k)**2.gt.0.0_WP) then
                        call setNumberOfPlanes(this%vf%liquid_gas_interface(i,j,k),2)
                        call setPlane(this%vf%liquid_gas_interface(i,j,k),0,[P11(i,j,k),P12(i,j,k),P13(i,j,k)],P14(i,j,k))
                        call setPlane(this%vf%liquid_gas_interface(i,j,k),1,[P21(i,j,k),P22(i,j,k),P23(i,j,k)],P24(i,j,k))
                     else if (Cr(i,j,k).gt.0.0_WP) then
                        call setAlignedCylinder(this%vf%liquid_gas_interface(i,j,k),1.0_WP,Cr(i,j,k),Cf(i,j,k))
                        datum(1) = Co1(i,j,k); datum(2) = Co2(i,j,k); datum(3) = Co3(i,j,k)
                        call setDatum(this%vf%liquid_gas_interface(i,j,k),datum)
                        ref1(1) = Cn11(i,j,k); ref1(2) = Cn12(i,j,k); ref1(3) = Cn13(i,j,k)
                        n1 = 0.0_WP; n2 = 0.0_WP; n3 = 0.0_WP; v1 = 0.0_WP
                        if (abs(ref1(1)) .ge. abs(ref1(2)) .and. abs(ref1(1)) .ge. abs(ref1(3))) then
                            n2 = ref1(1) / sqrt(ref1(2)**2 + ref1(1)**2)
                            n1 = (-n2 * ref1(2)) / ref1(1)
                            v1(1) = n1
                            v1(2) = n2
                            v1(3) = 0.0_WP
                        else if (abs(ref1(2)) .ge. abs(ref1(1)) .and. abs(ref1(2)) .ge. abs(ref1(3))) then
                            n1 = ref1(2) / sqrt(ref1(2)**2 + ref1(1)**2)
                            n2 = (-n1 * ref1(1)) / ref1(2)
                            v1(1) = n1
                            v1(2) = n2
                            v1(3) = 0.0_WP
                        else if (abs(ref1(3)) .ge. abs(ref1(1)) .and. abs(ref1(3)) .ge. abs(ref1(2))) then
                            n2 = ref1(3) / sqrt(ref1(2)**2 + ref1(3)**2)
                            n3 = (-n2 * ref1(2)) / ref1(3)
                            v1(1) = 0.0_WP
                            v1(2) = n2
                            v1(3) = n3
                        else
                            v1(1) = 0.0_WP; v1(2) = 0.0_WP; v1(3) = 0.0_WP
                        end if
                        ref3(1) = ref1(2) * v1(3) - ref1(3) * v1(2)
                        ref3(2) = ref1(3) * v1(1) - ref1(1) * v1(3)
                        ref3(3) = ref1(1) * v1(2) - ref1(2) * v1(1)
                        mag = sqrt(ref3(1)**2 + ref3(2)**2 + ref3(3)**2)
                        if (mag .ge. 1.0e-12) then
                           ref3 = ref3 / mag
                        end if
                        ref2(1) = ref3(2) * ref1(3) - ref3(3) * ref1(2)
                        ref2(2) = ref3(3) * ref1(1) - ref3(1) * ref1(3)
                        ref2(3) = ref3(1) * ref1(2) - ref3(2) * ref1(1)
                        mag = sqrt(ref2(1)**2 + ref2(2)**2 + ref2(3)**2)
                        if (mag .ge. 1.0e-12) then
                           ref2 = ref2 / mag
                        end if
                        call setReferenceFrame(this%vf%liquid_gas_interface(i,j,k),ref1,ref2,ref3)
                     else
                        call setNumberOfPlanes(this%vf%liquid_gas_interface(i,j,k),1)
                        call setPlane(this%vf%liquid_gas_interface(i,j,k),0,[P11(i,j,k),P12(i,j,k),P13(i,j,k)],P14(i,j,k))
                     end if
                     this%vf%det%recon_type(i,j,k)=RT(i,j,k)
                  end do
               end do
            end do
            call this%vf%sync_interface()
            deallocate(P11,P12,P13,P14,P21,P22,P23,P24,Cr,Co1,Co2,Co3,Cn11,Cn12,Cn13,RT,Cf)
            ! Reset moments
            call this%vf%reset_volume_moments()
            if (.not.this%vf%cfg%xper.and.this%vf%cfg%iproc.eq.1) then
               do k=this%vf%cfg%kmino_,this%vf%cfg%kmaxo_
                  do j=this%vf%cfg%jmino_,this%vf%cfg%jmaxo_
                     do i=this%vf%cfg%imino,this%vf%cfg%imin-1
                        this%vf%VF(i,j,k)=0.0_WP
                     end do
                  end do
               end do
            end if
            ! In X+
            if (.not.this%vf%cfg%xper.and.this%vf%cfg%iproc.eq.this%vf%cfg%npx) then
               do k=this%vf%cfg%kmino_,this%vf%cfg%kmaxo_
                  do j=this%vf%cfg%jmino_,this%vf%cfg%jmaxo_
                     do i=this%vf%cfg%imax+1,this%vf%cfg%imaxo
                        this%vf%VF(i,j,k)=0.0_WP
                     end do
                  end do
               end do
            end if
            ! In Y-
            if (.not.this%vf%cfg%yper.and.this%vf%cfg%jproc.eq.1) then
               do k=this%vf%cfg%kmino_,this%vf%cfg%kmaxo_
                  do j=this%vf%cfg%jmino,this%vf%cfg%jmin-1
                     do i=this%vf%cfg%imino_,this%vf%cfg%imaxo_
                        this%vf%VF(i,j,k)=0.0_WP
                     end do
                  end do
               end do
            end if
            ! In Y+
            if (.not.this%vf%cfg%yper.and.this%vf%cfg%jproc.eq.this%vf%cfg%npy) then
               do k=this%vf%cfg%kmino_,this%vf%cfg%kmaxo_
                  do j=this%vf%cfg%jmax+1,this%vf%cfg%jmaxo
                     do i=this%vf%cfg%imino_,this%vf%cfg%imaxo_
                        this%vf%VF(i,j,k)=0.0_WP
                     end do
                  end do
               end do
            end if
            ! In Z-
            if (.not.this%vf%cfg%zper.and.this%vf%cfg%kproc.eq.1) then
               do k=this%vf%cfg%kmino,this%vf%cfg%kmin-1
                  do j=this%vf%cfg%jmino_,this%vf%cfg%jmaxo_
                     do i=this%vf%cfg%imino_,this%vf%cfg%imaxo_
                        this%vf%VF(i,j,k)=0.0_WP
                     end do
                  end do
               end do
            end if
            ! In Z+
            if (.not.this%vf%cfg%zper.and.this%vf%cfg%kproc.eq.this%vf%cfg%npz) then
               do k=this%vf%cfg%kmax+1,this%vf%cfg%kmaxo
                  do j=this%vf%cfg%jmino_,this%vf%cfg%jmaxo_
                     do i=this%vf%cfg%imino_,this%vf%cfg%imaxo_
                        this%vf%VF(i,j,k)=0.0_WP
                     end do
                  end do
               end do
            end if
            ! Update the band
            call this%vf%update_band()
            ! Create discontinuous polygon mesh from IRL interface
            call this%vf%polygonalize_interface()
            ! Calculate distance from polygons
            !call this%vf%distance_from_polygon()
            ! Calculate subcell phasic volumes
            call this%vf%subcell_vol()
            ! Calculate curvature
            call this%vf%get_curvature()
            ! Also update time
            call this%df%pull(name='t' ,val=this%time%t )
            call this%df%pull(name='dt',val=this%time%dt)
            this%time%told=this%time%t-this%time%dt
            !this%time%dt=this%time%dtmax !< Force max timestep size anyway
         else
            ! Initialize to a ligament
            do k=this%vf%cfg%kmino_,this%vf%cfg%kmaxo_
               do j=this%vf%cfg%jmino_,this%vf%cfg%jmaxo_
                  do i=this%vf%cfg%imino_,this%vf%cfg%imaxo_
                     ! Set cube vertices
                     n=0
                     do sk=0,1
                        do sj=0,1
                           do si=0,1
                              n=n+1; cube_vertex(:,n)=[this%vf%cfg%x(i+si),this%vf%cfg%y(j+sj),this%vf%cfg%z(k+sk)]
                           end do
                        end do
                     end do
                     ! Call adaptive refinement code to get volume and barycenters recursively
                     vol=0.0_WP; area=0.0_WP; v_cent=0.0_WP; a_cent=0.0_WP
                     call cube_refine_vol(cube_vertex,vol,area,v_cent,a_cent,levelset_droplet,0.0_WP,amr_ref_lvl)
                     this%vf%VF(i,j,k)=vol/this%vf%cfg%vol(i,j,k)
                     if (this%vf%VF(i,j,k).ge.VFlo.and.this%vf%VF(i,j,k).le.VFhi) then
                        this%vf%Lbary(:,i,j,k)=v_cent
                        this%vf%Gbary(:,i,j,k)=([this%vf%cfg%xm(i),this%vf%cfg%ym(j),this%vf%cfg%zm(k)]-this%vf%VF(i,j,k)*this%vf%Lbary(:,i,j,k))/(1.0_WP-this%vf%VF(i,j,k))
                     else
                        this%vf%Lbary(:,i,j,k)=[this%vf%cfg%xm(i),this%vf%cfg%ym(j),this%vf%cfg%zm(k)]
                        this%vf%Gbary(:,i,j,k)=[this%vf%cfg%xm(i),this%vf%cfg%ym(j),this%vf%cfg%zm(k)]
                     end if
                  end do
               end do
            end do
            ! Update the band
            call this%vf%update_band()
            ! Perform interface reconstruction from VOF field
            call this%vf%build_interface()
            ! Set interface planes at the boundaries
            call this%vf%set_full_bcond()
            ! Create discontinuous polygon mesh from IRL interface
            call this%vf%polygonalize_interface()
            ! Calculate distance from polygons
            call this%vf%distance_from_polygon()
            ! Calculate subcell phasic volumes
            call this%vf%subcell_vol()
            ! Calculate curvature
            call this%vf%get_curvature()
            ! Reset moments to guarantee compatibility with interface reconstruction
            call this%vf%reset_volume_moments()
         end if
      end block create_and_initialize_vof
      
      
      ! Create an iterator for removing VOF at edges
      create_iterator: block
         this%vof_removal_layer=iterator(this%cfg,'VOF removal',vof_removal_layer_locator)
      end block create_iterator

      
      ! Create a multiphase flow solver with bconds
      create_flow_solver: block
         use mathtools,       only: Pi
         use param,           only: param_read
         use tpns_class,      only: dirichlet,clipped_neumann,bcond
         use hypre_str_class, only: pcg_pfmg2
         type(bcond), pointer :: mybc
         integer :: n,i,j,k      
         ! Create flow solver
         this%fs=tpns(cfg=this%cfg,name='Two-phase NS')
         ! Set fluid properties
         this%fs%rho_g=1.0_WP; call param_read('Density ratio',this%fs%rho_l)
         call param_read('Reynolds number',this%fs%visc_g); this%fs%visc_g=1.0_WP/this%fs%visc_g
         call param_read('Viscosity ratio',this%fs%visc_l); this%fs%visc_l=this%fs%visc_g*this%fs%visc_l
         call param_read('Weber number',this%fs%sigma); this%fs%sigma=1.0_WP/this%fs%sigma
         ! Define inflow boundary condition on the left
         call this%fs%add_bcond(name='inflow',type=dirichlet,face='x',dir=-1,canCorrect=.false.,locator=xm_locator)
         ! Define outflow boundary condition on the right
         call this%fs%add_bcond(name='outflow',type=clipped_neumann,face='x',dir=+1,canCorrect=.true.,locator=xp_locator)
         ! Configure pressure solver
         this%ps=hypre_str(cfg=this%cfg,name='Pressure',method=pcg_pfmg2,nst=7)
         this%ps%maxlevel=16
         call param_read('Pressure iteration',this%ps%maxit)
         call param_read('Pressure tolerance',this%ps%rcvg)
         ! Configure implicit velocity solver
         !this%vs=ddadi(cfg=this%cfg,name='Velocity',nst=7)
         ! Setup the solver
         call this%fs%setup(pressure_solver=this%ps)!,implicit_solver=this%vs)\
         ! Handle restart
         if (this%restarted) then
            call this%df%pull(name='U'  ,var=this%fs%U  )
            call this%df%pull(name='V'  ,var=this%fs%V  )
            call this%df%pull(name='W'  ,var=this%fs%W  )
            call this%df%pull(name='P'  ,var=this%fs%P  )
            call this%df%pull(name='Pjx',var=this%fs%Pjx)
            call this%df%pull(name='Pjy',var=this%fs%Pjy)
            call this%df%pull(name='Pjz',var=this%fs%Pjz)
         else
            ! Zero initial field
            this%fs%U=0.0_WP; this%fs%V=0.0_WP; this%fs%W=0.0_WP
            ! Apply convective velocity
            call this%fs%get_bcond('inflow',mybc)
            do n=1,mybc%itr%no_
               i=mybc%itr%map(1,n); j=mybc%itr%map(2,n); k=mybc%itr%map(3,n)
               this%fs%U(i,j,k)=1.0_WP
            end do
         end if
         ! Compute cell-centered velocity
         call this%fs%interp_vel(this%Ui,this%Vi,this%Wi)
         ! Compute divergence
         call this%fs%get_div()
      end block create_flow_solver
      
      ! Create a Lagrangian spray tracker
      create_lpt: block
         use param, only: param_read
         logical :: partfile_exists
         character(len=str_medium) :: timestamp

         call this%input%read('Restart from',timestamp,default='')
         ! Create the solver
         this%lp=lpt(cfg=this%cfg,name='spray')
         ! Get particle density from the flow solver
         this%lp%rho=this%fs%rho_l
         ! Turn off drag
         this%lp%drag_model='Schiller-Naumann'
         ! Initialize with zero particles
         call this%lp%resize(0)
         ! Get initial particle volume fraction
         call this%lp%update_VF()
         ! Get particle statistics
         call this%lp%get_max()
         if (this%restarted) then
            inquire(file='restart/part_'//trim(adjustl(timestamp)),exist=partfile_exists)
            ! If so, read it
            if (partfile_exists) call this%lp%read(filename='restart/part_'//trim(adjustl(timestamp)))
         end if
      end block create_lpt

      ! Prepare Lagrangian drop model
      prepare_transfer: block
         use messager,  only: die
         use filesys,  only: makedir,isdir
         integer :: ierr,iunit
         character(len=str_medium) :: filename
         ! ! Is transfer used?
         call this%input%read('Transfer drops',this%use_drop_transfer,default=.true.)
         call this%input%read('Transfer films',this%use_film_transfer,default=.true.)
         call this%input%read('Transfer ligaments',this%use_lig_transfer,default=.true.)
         call this%input%read('Transfer secondary',this%use_secondary,default=.true.)

         call this%vf%det%prepare_transfer(this%use_drop_transfer,this%use_film_transfer,this%use_lig_transfer,this%use_secondary,this%fs,this%lp)
      end block prepare_transfer

      ! Create surfmesh object for interface polygon output
      create_smesh: block
	     use irl_fortran_interface
	  	 integer :: i,j,k,nplane,np

         this%smesh=surfmesh(nvar=15,name='plic')
         this%smesh%varname(1)='lig'
		 this%smesh%varname(2)='ccl_lig'
		 this%smesh%varname(3)='recon_type'
		 this%smesh%varname(4)='thickness'
		 this%smesh%varname(5)='surface_area'
         this%smesh%varname(6)='curvature'
         this%smesh%varname(7)='tip'
         this%smesh%varname(8)='ml_class'
         this%smesh%varname(9)='snapped'
         this%smesh%varname(10)='r2p_tip'   ! 'tip' (7) is already the ligament edge sensor
         this%smesh%varname(11)='r2p_edge'
         this%smesh%varname(12)='r2p_unpinch'
         this%smesh%varname(13)='r2p_edge_topo'
         this%smesh%varname(14)='r2p_guard'
         this%smesh%varname(15)='r2p_one_plane'
         call this%vf%update_surfmesh(this%smesh)
	     this%smesh%var(1,:)=0.0_WP
	     this%smesh%var(2,:)=0.0_WP
	     this%smesh%var(3,:)=0.0_WP
		 this%smesh%var(4,:)=0.0_WP
		 this%smesh%var(5,:)=0.0_WP
       this%smesh%var(6,:)=0.0_WP
       this%smesh%var(7,:)=0.0_WP
       this%smesh%var(8,:)=0.0_WP
       this%smesh%var(9,:)=0.0_WP
       this%smesh%var(10,:)=0.0_WP
       this%smesh%var(11,:)=0.0_WP
       this%smesh%var(12,:)=0.0_WP
       this%smesh%var(13,:)=0.0_WP
       this%smesh%var(14,:)=0.0_WP
       this%smesh%var(15,:)=0.0_WP
		 call add_surfgrid_variable(this,this%smesh,1,real(this%vf%det%struct_type,WP)) 
           call add_surfgrid_variable(this,this%smesh,2,real(this%vf%det%ccl_recon%id,WP))    
		 call add_surfgrid_variable(this,this%smesh,3,real(this%vf%det%recon_type,WP))  
		 call add_surfgrid_variable(this,this%smesh,4,this%vf%thickness) 
		 call add_surfgrid_variable(this,this%smesh,5,this%vf%SD)  
       call add_surfgrid_variable(this,this%smesh,6,this%vf%curv) 
       call add_surfgrid_variable(this,this%smesh,7,real(this%vf%det%lig_edge_sensor,WP)) 
       call add_surfgrid_variable(this,this%smesh,8,this%vf%r2p_class)
       call add_surfgrid_variable(this,this%smesh,9,this%vf%r2p_snapped)
       call add_surfgrid_variable(this,this%smesh,10,this%vf%r2p_tip)
       call add_surfgrid_variable(this,this%smesh,11,this%vf%r2p_edge)
       call add_surfgrid_variable(this,this%smesh,12,this%vf%r2p_unpinch)
       call add_surfgrid_variable(this,this%smesh,13,this%vf%r2p_edge_topo)
       call add_surfgrid_variable(this,this%smesh,14,this%vf%r2p_guard)
       call add_surfgrid_variable(this,this%smesh,15,this%vf%r2p_one_plane)

      end block create_smesh

      ! Create partmesh object for Lagrangian particle output
      if (this%use_drop_transfer.or.this%use_film_transfer.or.this%use_lig_transfer) then
         create_pmesh: block
            integer :: i
            ! Include an extra variable for droplet diameter
            this%pmesh=partmesh(nvar=2,nvec=1,name='lpt')
            this%pmesh%varname(1)='diameter'
            this%pmesh%varname(2)='id'
            this%pmesh%vecname(1)='velocity'
            ! Transfer particles to pmesh
            call this%lp%update_partmesh(this%pmesh)
            ! Also populate diameter variable
            do i=1,this%lp%np_
               this%pmesh%var(1,i)=this%lp%p(i)%d
               this%pmesh%var(2,i)=this%lp%p(i)%id
               this%pmesh%vec(:,1,i)=this%lp%p(i)%vel
            end do
            ! if (this%lp%np.eq.0.and.this%lp%cfg%amRoot) then
            !    this%pmesh%var(1,:)=0.0_WP
            !    this%pmesh%var(2,:)=0
            ! end if
         end block create_pmesh
      end if

      ! Add Ensight output
      create_ensight: block
         use param, only: param_read
         ! Create Ensight output from cfg
         this%ens_out=ensight(cfg=this%cfg,name='ligament')
         ! Create event for Ensight output
         this%ens_evt=event(time=this%time,name='Ensight output')
         call param_read('Ensight output period',this%ens_evt%tper)
         ! Add variables to output
         call this%ens_out%add_vector('velocity',this%Ui,this%Vi,this%Wi)
         call this%ens_out%add_scalar('VOF',this%vf%VF)
         !call this%ens_out%add_scalar('curvature',this%vf%curv)
         call this%ens_out%add_scalar('pressure',this%fs%P)
         !call this%ens_out%add_scalar('thin_sensor',this%vf%thin_sensor)
         !call this%ens_out%add_scalar('edge_sensor',this%vf%edge_sensor)
         !call this%ens_out%add_vector('edge_normal',this%resU,this%resV,this%resW)
         call this%ens_out%add_surface('plic',this%smesh)
         call this%ens_out%add_particle('spray',this%pmesh)
         ! Output to ensight
         if (this%ens_evt%occurs()) call this%ens_out%write_data(this%time%t)

      end block create_ensight
      

      ! Create a monitor file
      create_monitor: block
         ! Prepare some info about fields
         call this%fs%get_cfl(this%time%dt,this%time%cfl)
         call this%fs%get_max()
         call this%vf%get_max()
         ! Create simulation monitor
         this%mfile=monitor(this%fs%cfg%amRoot,'simulation_atom')
         call this%mfile%add_column(this%time%n,'Timestep number')
         call this%mfile%add_column(this%time%t,'Time')
         call this%mfile%add_column(this%time%dt,'Timestep size')
         call this%mfile%add_column(this%time%cfl,'Maximum CFL')
         call this%mfile%add_column(this%fs%Umax,'Umax')
         call this%mfile%add_column(this%fs%Vmax,'Vmax')
         call this%mfile%add_column(this%fs%Wmax,'Wmax')
         call this%mfile%add_column(this%fs%Pmax,'Pmax')
         call this%mfile%add_column(this%vf%VFmax,'VOF maximum')
         call this%mfile%add_column(this%vf%VFmin,'VOF minimum')
         call this%mfile%add_column(this%vf%VFint,'VOF integral')
         call this%mfile%add_column(this%vf%flotsam_error,'Flotsam error')
         call this%mfile%add_column(this%vf%thinstruct_error,'Film error')
         call this%mfile%add_column(this%vf%SDint,'SD integral')
         call this%mfile%add_column(this%fs%divmax,'Maximum divergence')
         call this%mfile%add_column(this%fs%psolv%it,'Pressure iteration')
         call this%mfile%add_column(this%fs%psolv%rerr,'Pressure error')
         call this%mfile%write()
         ! Create CFL monitor
         this%cflfile=monitor(this%fs%cfg%amRoot,'cfl_atom')
         call this%cflfile%add_column(this%time%n,'Timestep number')
         call this%cflfile%add_column(this%time%t,'Time')
         call this%cflfile%add_column(this%fs%CFLst,'STension CFL')
         call this%cflfile%add_column(this%fs%CFLc_x,'Convective xCFL')
         call this%cflfile%add_column(this%fs%CFLc_y,'Convective yCFL')
         call this%cflfile%add_column(this%fs%CFLc_z,'Convective zCFL')
         call this%cflfile%add_column(this%fs%CFLv_x,'Viscous xCFL')
         call this%cflfile%add_column(this%fs%CFLv_y,'Viscous yCFL')
         call this%cflfile%add_column(this%fs%CFLv_z,'Viscous zCFL')
         call this%cflfile%write()
         ! Create a droplet mean/median monitor
         if (this%use_drop_transfer.or.this%use_film_transfer.or.this%use_lig_transfer) then
            this%dropfile=monitor(amroot=this%lp%cfg%amRoot,name='dropstats')
            call this%dropfile%add_column(this%time%n,'Timestep number')
            call this%dropfile%add_column(this%time%t,'Time')
            call this%dropfile%add_column(this%time%dt,'Timestep size')
            call this%dropfile%add_column(this%lp%np,'Droplet number')
            call this%dropfile%add_column(this%lp%dmean,'d10')
            call this%dropfile%write()
         end if

      end block create_monitor
      
   end subroutine init
   
   
   !> Take one time step
   subroutine step(this)
      use tpns_class, only: arithmetic_visc,harmonic_visc
      implicit none
      class(ligament), intent(inout) :: this

      ! Increment time
      call this%fs%get_cfl(this%time%dt,this%time%cfl)
      call this%time%adjust_dt()
      call this%time%increment()
      this%vf%r2p_dump_step=this%time%n   ! time step for the r2p_plic_cells dump

      if (this%use_drop_transfer.or.this%use_film_transfer.or.this%use_lig_transfer) then
         this%resU=this%fs%rho_g; this%resV=this%fs%visc_g
         call this%lp%advance(dt=this%time%dt,U=this%fs%U,V=this%fs%V,W=this%fs%W,rho=this%resU,visc=this%resV)
      end if
      
      ! Remember old VOF
      this%vf%VFold=this%vf%VF

      ! Remember old velocity
      this%fs%Uold=this%fs%U
      this%fs%Vold=this%fs%V
      this%fs%Wold=this%fs%W
      
      ! Prepare old staggered density (at n)
      call this%fs%get_olddensity(vf=this%vf)
         
      ! VOF solver step
      call this%vf%advance(dt=this%time%dt,U=this%fs%U,V=this%fs%V,W=this%fs%W)
      
      ! Prepare new staggered viscosity (at n+1)
      call this%fs%get_viscosity(vf=this%vf,strat=arithmetic_visc)
      
      ! Perform sub-iterations
      do while (this%time%it.le.this%time%itmax)
         ! Build mid-time velocity
         this%fs%U=0.5_WP*(this%fs%U+this%fs%Uold)
         this%fs%V=0.5_WP*(this%fs%V+this%fs%Vold)
         this%fs%W=0.5_WP*(this%fs%W+this%fs%Wold)

         ! Preliminary mass and momentum transport step at the interface
         call this%fs%prepare_advection_upwind(dt=this%time%dt)

         ! Explicit calculation of drho*u/dt from NS
         call this%fs%get_dmomdt(this%resU,this%resV,this%resW)

         ! Assemble explicit residual
         this%resU=-2.0_WP*this%fs%rho_U*this%fs%U+(this%fs%rho_Uold+this%fs%rho_U)*this%fs%Uold+this%time%dt*this%resU
         this%resV=-2.0_WP*this%fs%rho_V*this%fs%V+(this%fs%rho_Vold+this%fs%rho_V)*this%fs%Vold+this%time%dt*this%resV
         this%resW=-2.0_WP*this%fs%rho_W*this%fs%W+(this%fs%rho_Wold+this%fs%rho_W)*this%fs%Wold+this%time%dt*this%resW   
         
         ! Form implicit residuals
         !call this%fs%solve_implicit(this%time%dt,this%resU,this%resV,this%resW)

         ! Apply these residuals
         this%fs%U=2.0_WP*this%fs%U-this%fs%Uold+this%resU/this%fs%rho_U
         this%fs%V=2.0_WP*this%fs%V-this%fs%Vold+this%resV/this%fs%rho_V
         this%fs%W=2.0_WP*this%fs%W-this%fs%Wold+this%resW/this%fs%rho_W

         ! Solve Poisson equation
         call this%fs%update_laplacian()
         !call this%fs%update_laplacian(pinpoint=[this%fs%cfg%imin,this%fs%cfg%jmin,this%fs%cfg%kmin])
         call this%fs%correct_mfr()
         call this%fs%get_div()

         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         !seg fault line
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         !call this%fs%add_surface_tension_jump(dt=this%time%dt,div=this%fs%div,vf=this%vf)
         call this%fs%add_surface_tension_jump_thin(dt=this%time%dt,div=this%fs%div,vf=this%vf)

         this%fs%psolv%rhs=-this%fs%cfg%vol*this%fs%div/this%time%dt

         !if (this%cfg%amRoot) this%fs%psolv%rhs(this%cfg%imin,this%cfg%jmin,this%cfg%kmin)=0.0_WP
         this%fs%psolv%sol=0.0_WP
         call this%fs%psolv%solve()
         call this%fs%shift_p(this%fs%psolv%sol)
         ! Correct velocity
         call this%fs%get_pgrad(this%fs%psolv%sol,this%resU,this%resV,this%resW)
         this%fs%P=this%fs%P+this%fs%psolv%sol
         this%fs%U=this%fs%U-this%time%dt*this%resU/this%fs%rho_U
         this%fs%V=this%fs%V-this%time%dt*this%resV/this%fs%rho_V
         this%fs%W=this%fs%W-this%time%dt*this%resW/this%fs%rho_W
         ! Increment sub-iteration counter
         this%time%it=this%time%it+1
         
      end do
      
      ! Recompute interpolated velocity and divergence
      call this%fs%interp_vel(this%Ui,this%Vi,this%Wi)
      call this%fs%get_div()

      call this%vf%det%attempt_transfer(this%time%dt)
      
      ! Remove VOF at edge of domain
      remove_vof: block
         integer :: n
         do n=1,this%vof_removal_layer%no_
            this%vf%VF(this%vof_removal_layer%map(1,n),this%vof_removal_layer%map(2,n),this%vof_removal_layer%map(3,n))=0.0_WP
         end do
      end block remove_vof

     !call this%tm%spray_statistics()

      ! Output to ensight
      if (this%ens_evt%occurs()) then
         ! Update surfmesh object
				update_smesh: block
					use irl_fortran_interface
					integer :: i,j,k,nplane,np
					! Transfer polygons to smesh
					call this%vf%update_surfmesh(this%smesh)
					! Also populate nplane variable
					this%smesh%var(1,:)=0.0_WP
					this%smesh%var(2,:)=0.0_WP
					this%smesh%var(3,:)=0.0_WP
					this%smesh%var(4,:)=0.0_WP
					this%smesh%var(5,:)=0.0_WP
               this%smesh%var(6,:)=0.0_WP
               this%smesh%var(7,:)=0.0_WP
               this%smesh%var(8,:)=0.0_WP
               this%smesh%var(9,:)=0.0_WP
               this%smesh%var(10,:)=0.0_WP
               this%smesh%var(11,:)=0.0_WP
               this%smesh%var(12,:)=0.0_WP
               this%smesh%var(13,:)=0.0_WP
               this%smesh%var(14,:)=0.0_WP
               this%smesh%var(15,:)=0.0_WP
					call add_surfgrid_variable(this,this%smesh,1,real(this%vf%det%struct_type,WP)) 
               call add_surfgrid_variable(this,this%smesh,2,real(this%vf%det%ccl_recon%id,WP))    
					call add_surfgrid_variable(this,this%smesh,3,real(this%vf%det%recon_type,WP))  
					call add_surfgrid_variable(this,this%smesh,4,this%vf%thickness)  
					call add_surfgrid_variable(this,this%smesh,5,this%vf%SD)  
               call add_surfgrid_variable(this,this%smesh,6,this%vf%curv) 
               call add_surfgrid_variable(this,this%smesh,7,real(this%vf%det%lig_edge_sensor,WP)) 
               call add_surfgrid_variable(this,this%smesh,8,this%vf%r2p_class)
               call add_surfgrid_variable(this,this%smesh,9,this%vf%r2p_snapped)
               call add_surfgrid_variable(this,this%smesh,10,this%vf%r2p_tip)
               call add_surfgrid_variable(this,this%smesh,11,this%vf%r2p_edge)
               call add_surfgrid_variable(this,this%smesh,12,this%vf%r2p_unpinch)
               call add_surfgrid_variable(this,this%smesh,13,this%vf%r2p_edge_topo)
               call add_surfgrid_variable(this,this%smesh,14,this%vf%r2p_guard)
               call add_surfgrid_variable(this,this%smesh,15,this%vf%r2p_one_plane)
				end block update_smesh
         ! Transfer edge normal data
         !this%resU=this%vf%edge_normal(1,:,:,:)
         !this%resV=this%vf%edge_normal(2,:,:,:)
         !this%resW=this%vf%edge_normal(3,:,:,:)
         ! Update partmesh object
         if (this%use_drop_transfer.or.this%use_film_transfer.or.this%use_lig_transfer) then
            update_pmesh: block
               integer :: i
               ! Transfer particles to pmesh
               call this%lp%update_partmesh(this%pmesh)
               ! Also populate diameter variable
               do i=1,this%lp%np_
                  this%pmesh%var(1,i)=this%lp%p(i)%d
                  this%pmesh%var(2,i)=this%lp%p(i)%id
                  this%pmesh%vec(:,1,i)=this%lp%p(i)%vel
               end do
               ! if (this%lp%np.eq.0.and.this%lp%cfg%amRoot) then
               !    this%pmesh%var(1,:)=0.0_WP
               !    this%pmesh%var(2,:)=0
               ! end if
            end block update_pmesh
         end if
         ! Perform ensight output
         call this%ens_out%write_data(this%time%t)
      end if

      ! Perform and output monitoring
      call this%fs%get_max()
      call this%vf%get_max()
      call this%mfile%write()
      call this%cflfile%write()
   
      if (this%save_evt%occurs()) then
         save_restart: block
            use irl_fortran_interface
            use string, only: str_medium
            character(len=str_medium) :: timestamp
            real(WP), dimension(:,:,:), allocatable :: P11,P12,P13,P14
            real(WP), dimension(:,:,:), allocatable :: P21,P22,P23,P24
            real(WP), dimension(:,:,:), allocatable :: Cr,Co1,Co2,Co3,Cn11,Cn12,Cn13,RT,Cf
            integer :: i,j,k
            real(WP), dimension(4) :: plane
            real(WP), dimension(3) :: aligned_cyl
            real(WP), dimension(3) :: datum
            real(WP), dimension(9) :: ref
            ! Handle IRL data
            allocate(P11(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p11=0.0_WP
            allocate(P12(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p12=0.0_WP
            allocate(P13(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p13=0.0_WP
            allocate(P14(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p14=0.0_WP
            allocate(P21(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p21=0.0_WP
            allocate(P22(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p22=0.0_WP
            allocate(P23(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p23=0.0_WP
            allocate(P24(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); p24=0.0_WP
            allocate(Cr(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Cr=0.0_WP
            allocate(Co1(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Co1=0.0_WP
            allocate(Co2(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Co2=0.0_WP
            allocate(Co3(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Co3=0.0_WP
            allocate(Cn11(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Cn11=0.0_WP
            allocate(Cn12(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Cn12=0.0_WP
            allocate(Cn13(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Cn13=0.0_WP
            allocate(RT(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); RT=0.0_WP
            allocate(Cf(this%cfg%imino_:this%cfg%imaxo_,this%cfg%jmino_:this%cfg%jmaxo_,this%cfg%kmino_:this%cfg%kmaxo_)); Cf=0.0_WP
            do k=this%vf%cfg%kmino_,this%vf%cfg%kmaxo_
               do j=this%vf%cfg%jmino_,this%vf%cfg%jmaxo_
                  do i=this%vf%cfg%imino_,this%vf%cfg%imaxo_
                     RT(i,j,k)=2!this%vf%det%recon_type(i,j,k)
                     if (this%vf%det%recon_type(i,j,k).eq.1) then
                        aligned_cyl=getAlignedCylinder(this%vf%liquid_gas_interface(i,j,k))
                        datum=getDatum(this%vf%liquid_gas_interface(i,j,k))
                        ref=getReferenceFrame(this%vf%liquid_gas_interface(i,j,k))
                        Cr(i,j,k)=aligned_cyl(1)
                        Cf(i,j,k)=aligned_cyl(3)
                        Co1(i,j,k)=datum(1)
                        Co2(i,j,k)=datum(2)
                        Co3(i,j,k)=datum(3)
                        Cn11(i,j,k)=ref(1)
                        Cn12(i,j,k)=ref(2)
                        Cn13(i,j,k)=ref(3)
                     else
                        ! First plane
                        plane=getPlane(this%vf%liquid_gas_interface(i,j,k),0)
                        P11(i,j,k)=plane(1); P12(i,j,k)=plane(2); P13(i,j,k)=plane(3); P14(i,j,k)=plane(4)
                        ! Second plane
                        plane=0.0_WP
                        if (getNumberOfPlanes(this%vf%liquid_gas_interface(i,j,k)).eq.2) plane=getPlane(this%vf%liquid_gas_interface(i,j,k),1)
                        P21(i,j,k)=plane(1); P22(i,j,k)=plane(2); P23(i,j,k)=plane(3); P24(i,j,k)=plane(4)
                     end if
                  end do
               end do
            end do
            ! Prefix for files
            write(timestamp,'(es12.5)') this%time%t
            ! Populate df and write it
            call this%df%push(name='t'  ,val=this%time%t )
            call this%df%push(name='dt' ,val=this%time%dt)
            call this%df%push(name='U'  ,var=this%fs%U   )
            call this%df%push(name='V'  ,var=this%fs%V   )
            call this%df%push(name='W'  ,var=this%fs%W   )
            call this%df%push(name='P'  ,var=this%fs%P   )
            call this%df%push(name='Pjx',var=this%fs%Pjx )
            call this%df%push(name='Pjy',var=this%fs%Pjy )
            call this%df%push(name='Pjz',var=this%fs%Pjz )
            call this%df%push(name='P11',var=P11         )
            call this%df%push(name='P12',var=P12         )
            call this%df%push(name='P13',var=P13         )
            call this%df%push(name='P14',var=P14         )
            call this%df%push(name='P21',var=P21         )
            call this%df%push(name='P22',var=P22         )
            call this%df%push(name='P23',var=P23         )
            call this%df%push(name='P24',var=P24         )
            call this%df%push(name='Cr',var=Cr           )
            call this%df%push(name='Cf',var=Cf           )
            call this%df%push(name='Co1',var=Co1         )
            call this%df%push(name='Co2',var=Co2         )
            call this%df%push(name='Co3',var=Co3         )
            call this%df%push(name='C11',var=Cn11         )
            call this%df%push(name='C12',var=Cn12         )
            call this%df%push(name='C13',var=Cn13         )
            call this%df%push(name='RT',var=RT         )

            call this%df%write(fdata='restart/data_'//trim(adjustl(timestamp)))
            ! Also output particles
            if (this%lp%np.gt.0) call this%lp%write(filename='restart/part_'//trim(adjustl(timestamp)))
            ! Deallocate
            deallocate(P11,P12,P13,P14,P21,P22,P23,P24,Cr,Cf,Co1,Co2,Co3,Cn11,Cn12,Cn13,RT)
         end block save_restart
      end if

   end subroutine step
   

   !> Finalize nozzle simulation
   subroutine final(this)
      implicit none
      class(ligament), intent(inout) :: this
      
      ! Deallocate work arrays
      deallocate(this%resU,this%resV,this%resW,this%Ui,this%Vi,this%Wi)
      
   end subroutine final
   
   
   !> Function that localizes the x- boundary
   function xm_locator(pg,i,j,k) result(isIn)
      use pgrid_class, only: pgrid
      class(pgrid), intent(in) :: pg
      integer, intent(in) :: i,j,k
      logical :: isIn
      isIn=.false.
      if (i.eq.pg%imin) isIn=.true.
   end function xm_locator


   !> Function that localizes the x+ boundary
   function xp_locator(pg,i,j,k) result(isIn)
      use pgrid_class, only: pgrid
      class(pgrid), intent(in) :: pg
      integer, intent(in) :: i,j,k
      logical :: isIn
      isIn=.false.
      if (i.eq.pg%imax+1) isIn=.true.
   end function xp_locator
   
   
   !> Function that localizes region of VOF removal
   function vof_removal_layer_locator(pg,i,j,k) result(isIn)
      use pgrid_class, only: pgrid
      class(pgrid), intent(in) :: pg
      integer, intent(in) :: i,j,k
      logical :: isIn
      isIn=.false.
      if (i.ge.pg%imax-4) isIn=.true.
   end function vof_removal_layer_locator

	!> Make a surface scalar variable
	subroutine add_surfgrid_variable(this,smesh,var_index,A)
		use irl_fortran_interface
		use vfs_class,only: VFhi,VFlo
		implicit none
                class(ligament), intent(inout) :: this
		class(surfmesh), intent(inout) :: smesh
		integer, intent(in) :: var_index
		real(WP), dimension(this%vf%cfg%imino_:this%vf%cfg%imaxo_,this%vf%cfg%jmino_:this%vf%cfg%jmaxo_,this%vf%cfg%kmino_:this%vf%cfg%kmaxo_), intent(in) :: A 
		integer :: i,j,k,n,shape,np,nplane,nbt
		
		! Fill out arrays
		if ((smesh%nPoly+smesh%nBezierTri).gt.0) then
		   np=0; nbt = 0
		   ! Start with quadratic surfaces
		   do k=this%vf%cfg%kmin_,this%vf%cfg%kmax_
			  do j=this%vf%cfg%jmin_,this%vf%cfg%jmax_
				 do i=this%vf%cfg%imin_,this%vf%cfg%imax_
					shape=getNumberOfTriangles(this%vf%interface_mixed_surface(i,j,k))
					if (shape.gt.0) then
					   do n=1,shape
						  nbt=nbt+1
						  smesh%var(var_index,nbt)=A(i,j,k)
					   end do
					end if
				 end do
			  end do
		   end do
		   ! Then do planes
		   do k=this%vf%cfg%kmin_,this%vf%cfg%kmax_
			  do j=this%vf%cfg%jmin_,this%vf%cfg%jmax_
				 do i=this%vf%cfg%imin_,this%vf%cfg%imax_
					do nplane=1,getNumberOfPlanes(this%vf%liquid_gas_interface(i,j,k))
					   shape=getNumberOfVertices(this%vf%interface_polygon(nplane,i,j,k))
					   if (shape.gt.0) then
						  ! Increment polygon counter
						  np=np+1
						  ! Set nplane variable
						  smesh%var(var_index,nbt+np)=A(i,j,k)
					   end if
					end do
				 end do
			  end do
		   end do
		else
		   smesh%var(var_index,1)=1
		end if      
		
	 end subroutine add_surfgrid_variable

   
   
end module ligament_class