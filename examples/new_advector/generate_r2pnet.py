#!/usr/bin/env python3

# Generates the r2pnet.f90 file for a given Pytorch model.
#
# Fortran counterpart of generate_r2pnet_c.py, written in the style of
# generate_plicnet.py / plicnet.f90. The network evaluated here is numerically
# identical to the one in r2pnet.h; only the layout and interface differ:
#
#   * Weights are emitted TRANSPOSED. r2pnet.h stores lay_weight[out][in] and
#     loops explicitly; Fortran stores lay_weight(in,out) and uses matmul, so
#     lay_weight(i,j) here == lay_weight[j][i] in the C header.
#
#   * get_normals returns the six outputs as two 3-vectors, normal1 = out(1:3)
#     and normal2 = out(4:6), matching normal[0..2] and normal[3..5] in the C.
#
#   * Weight literals carry a d exponent so they are double precision. Bare
#     e-notation in a DATA statement is DEFAULT REAL, which silently truncates
#     to ~7 digits and shifts every network output by ~1e-8.
#
#   * reflect_moments initializes direction and direction2 to 0. The C version
#     seeds them from the values passed in, so if no branch fires it returns
#     whatever the caller happened to supply. The Fortran behaviour matches
#     plicnet.f90 and is the intended one.

import torch
import numpy as np

torch.set_default_dtype(torch.float64)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1900)
model = torch.jit.load('./model_r2p_BEST_FLAT.pt')


def fortran_double(v):
    """Shortest literal that round-trips to the same double, with a d exponent.

    This matters. A DATA literal written as 0.17640523 has no kind suffix, so
    Fortran treats it as DEFAULT REAL - single precision - and only then widens
    it to real(WP). That silently costs ~7 significant digits: a network built
    that way disagrees with r2pnet.h by ~1e-8 on every output. The d exponent
    makes the literal double precision, and repr() guarantees it reads back as
    the exact bit pattern PyTorch held.

    Note plicnet.f90 writes bare e-notation literals and so carries this loss.
    """
    s = repr(float(v))
    if 'e' in s or 'E' in s:
        return s.replace('e', 'd').replace('E', 'd')
    return s + 'd0'


def data_statement(name, index_expr, values, count, file):
    """Emit a Fortran DATA statement, wrapping on commas with & continuations."""
    values_str = ', '.join(fortran_double(v) for v in np.asarray(values).ravel())
    chunks, max_len = [], 1900
    while len(values_str) > max_len:
        cut = values_str.rfind(',', 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(values_str[:cut + 1])
        values_str = values_str[cut + 1:].lstrip()
    chunks.append(values_str)
    print(f"   DATA ({name}({index_expr}), idx=1, {count}) /&", file=file)
    for n, chunk in enumerate(chunks):
        tail = "&" if n < len(chunks) - 1 else "/"
        print(f"   {chunk}{tail}", file=file)


file = open("r2pnet.f90", "w")

print("!> R2P-Net File", file=file)
print("!> Provides the architecture for the neural network and the weights/biases", file=file)
print("!> Use generate_r2pnet.py in NGA2/tools/scripts/r2pnet to generate this file for a given Pytorch model", file=file)
print("module r2pnet", file=file)
print("   use precision, only: WP", file=file)
print("   implicit none", file=file)
print("   integer :: idx", file=file)

# ---- Collect parameters -------------------------------------------------
param_info = []
count = 0
for param in model.parameters():
    count += 1
    if count % 2 != 0:
        name = "lay" + str(int(count / 2) + 1) + "_weight"
        param_info.append({'name': name, 'type': 'weight', 'data': param.detach().numpy()})
    else:
        name = "lay" + str(int((count - 1) / 2) + 1) + "_bias"
        param_info.append({'name': name, 'type': 'bias', 'data': param.detach().numpy()})
num_layers = int(count / 2)

# ---- Declarations -------------------------------------------------------
for info in param_info:
    data = info['data']
    if info['type'] == 'weight':
        # Transposed relative to the C header: (in_features, out_features)
        print(f"   real(WP), dimension({data.shape[1]},{data.shape[0]}), save :: {info['name']}", file=file)
    else:
        print(f"   real(WP), dimension({data.shape[0]}), save :: {info['name']}", file=file)

# ---- DATA statements ----------------------------------------------------
for info in param_info:
    name, data = info['name'], info['data']
    if info['type'] == 'weight':
        out_features, in_features = data.shape
        # Column j of the Fortran array is output neuron j's input weights,
        # which is row j of the PyTorch weight matrix.
        for m in range(out_features):
            data_statement(name, f"idx, {m + 1}", data[m, :], in_features, file)
    else:
        data_statement(name, "idx", data, data.shape[0], file)

# ---- Forward pass -------------------------------------------------------
out_size = param_info[2 * (num_layers - 1)]['data'].shape[0]
hidden = param_info[0]['data'].shape[0]

print("", file=file)
print("   contains", file=file)
print("   subroutine get_normals(moments,normal1,normal2)", file=file)
print("      implicit none", file=file)
print("      real(WP), dimension(:), intent(in) :: moments  !< Needs to be of size 189", file=file)
print("      real(WP), dimension(:), intent(out) :: normal1 !< Needs to be of size 3", file=file)
print("      real(WP), dimension(:), intent(out) :: normal2 !< Needs to be of size 3", file=file)
print(f"      real(WP), dimension({hidden}) :: tmparr", file=file)
print(f"      real(WP), dimension({out_size}) :: outarr", file=file)
for i in range(num_layers):
    src_arr = "moments" if i == 0 else "tmparr"
    dst_arr = "outarr" if i == num_layers - 1 else "tmparr"
    expr = f"matmul({src_arr},lay{i + 1}_weight)+lay{i + 1}_bias"
    # ReLU on every layer but the last
    if i < num_layers - 1:
        print(f"      {dst_arr}=max(0.0_WP,{expr})", file=file)
    else:
        print(f"      {dst_arr}={expr}", file=file)
print("      normal1=outarr(1:3)", file=file)
print("      normal2=outarr(4:6)", file=file)
print("   end subroutine", file=file)

# ---- Reflection routines ------------------------------------------------
# Identical to plicnet.f90 (and to the C header) apart from the direction
# initialization noted at the top of this file.
reflect_subroutines = r"""   subroutine reflect_moments(moments,center,direction,direction2)
      implicit none
      real(WP), dimension(0:), intent(inout) :: moments !< Needs to be of size (0:188)
      real(WP), dimension(0:), intent(in) :: center     !< Needs to be of size (0:2)
      integer, intent(out) :: direction, direction2
      real(WP), dimension(0:2) :: new_center
      real(WP) :: temp
      direction=0
      direction2=0
      new_center = center
      if (abs(new_center(0)).le.1e-12) new_center(0)=0
      if (abs(new_center(1)).le.1e-12) new_center(1)=0
      if (abs(new_center(2)).le.1e-12) new_center(2)=0
      if (new_center(0).lt.0.and.new_center(1).ge.0.and.new_center(2).ge.0) then
         direction=1
         call reflect_moments_x(moments)
         new_center(0) = -new_center(0)
      else if (new_center(0).ge.0.and.new_center(1).lt.0.and.new_center(2).ge.0) then
         direction=2
         call reflect_moments_y(moments)
         new_center(1) = -new_center(1)
      else if (new_center(0).ge.0.and.new_center(1).ge.0.and.new_center(2).lt.0) then
         direction=3
         call reflect_moments_z(moments)
         new_center(2) = -new_center(2)
      else if (new_center(0).lt.0.and.new_center(1).lt.0.and.new_center(2).ge.0) then
         direction=4
         call reflect_moments_x(moments)
         call reflect_moments_y(moments)
         new_center(0) = -new_center(0)
         new_center(1) = -new_center(1)
      else if (new_center(0).lt.0.and.new_center(1).ge.0.and.new_center(2).lt.0) then
         direction=5
         call reflect_moments_x(moments)
         call reflect_moments_z(moments)
         new_center(0) = -new_center(0)
         new_center(2) = -new_center(2)
      else if (new_center(0).ge.0.and.new_center(1).lt.0.and.new_center(2).lt.0) then
         direction=6
         call reflect_moments_y(moments)
         call reflect_moments_z(moments)
         new_center(1) = -new_center(1)
         new_center(2) = -new_center(2)
      else if (new_center(0).lt.0.and.new_center(1).lt.0.and.new_center(2).lt.0) then
         direction=7
         call reflect_moments_x(moments)
         call reflect_moments_y(moments)
         call reflect_moments_z(moments)
         new_center(0) = -new_center(0)
         new_center(1) = -new_center(1)
         new_center(2) = -new_center(2)
      end if

      if (abs(new_center(0)-new_center(1)).le.1e-12.and.(new_center(0)-new_center(2)).gt.1e-12) then
         direction2=0
      else if (abs(new_center(1)-new_center(2)).le.1e-12.and.(new_center(0)-new_center(1)).gt.1e-12) then
         direction2=0
      else if (abs(new_center(0)-new_center(1)).le.1e-12.and.(new_center(2)-new_center(0)).gt.1e-12) then
         direction2=3
         call reflect_moments_xz(moments)
         temp = new_center(0)
         new_center(0) = new_center(2)
         new_center(2) = temp
      else if (abs(new_center(0)-new_center(2)).le.1e-12.and.(new_center(1)-new_center(0)).gt.1e-12) then
         direction2=1
         call reflect_moments_xy(moments)
         temp = new_center(0)
         new_center(0) = new_center(1)
         new_center(1) = temp
      else if (abs(new_center(0)-new_center(2)).le.1e-12.and.(new_center(0)-new_center(1)).gt.1e-12) then
         direction2=2
         call reflect_moments_yz(moments)
         temp = new_center(1)
         new_center(1) = new_center(2)
         new_center(2) = temp
      else if (abs(new_center(1)-new_center(2)).le.1e-12.and.(new_center(1)-new_center(0)).gt.1e-12) then
         direction2=3
         call reflect_moments_xz(moments)
         temp = new_center(0)
         new_center(0) = new_center(2)
         new_center(2) = temp
      else if (new_center(1).gt.new_center(0).and.new_center(0).ge.new_center(2)) then
         direction2=1
         call reflect_moments_xy(moments)
         temp = new_center(0)
         new_center(0) = new_center(1)
         new_center(1) = temp
      else if (new_center(2).gt.new_center(1).and.new_center(0).ge.new_center(2)) then
         direction2=2
         call reflect_moments_yz(moments)
         temp = new_center(1)
         new_center(1) = new_center(2)
         new_center(2) = temp
      else if (new_center(2).gt.new_center(1).and.new_center(1).ge.new_center(0)) then
         direction2=3
         call reflect_moments_xz(moments)
         temp = new_center(0)
         new_center(0) = new_center(2)
         new_center(2) = temp
      else if (new_center(1).gt.new_center(0)) then
         direction2=4
         call reflect_moments_xy(moments)
         call reflect_moments_yz(moments)
         temp = new_center(0)
         new_center(0) = new_center(1)
         new_center(1) = temp
         temp = new_center(1)
         new_center(1) = new_center(2)
         new_center(2) = temp
      else if (new_center(2).gt.new_center(1)) then
         direction2=5
         call reflect_moments_xy(moments)
         call reflect_moments_xz(moments)
         temp = new_center(0)
         new_center(0) = new_center(1)
         new_center(1) = temp
         temp = new_center(0)
         new_center(0) = new_center(2)
         new_center(2) = temp
      end if
   end subroutine reflect_moments
   subroutine reflect_moments_x(moments)
      implicit none
      real(WP), dimension(0:), intent(inout) :: moments !< Needs to be of size (0:188)
      integer :: i,j,k,n
      real(WP) :: temp
      do k=0,2
         do j=0,2
            do i=0,2
               if (i.eq.0) then
                  do n=0,6
                     if (n.eq.1.or.n.eq.4) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=-moments(7*(2*9+j*3+k)+n)
                        moments(7*(2*9+j*3+k)+n)=-temp
                     else
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=+moments(7*(2*9+j*3+k)+n)
                        moments(7*(2*9+j*3+k)+n)=+temp
                     end if
                  end do
               else if (i.eq.1) then
                  moments(7*(i*9+j*3+k)+1)=-moments(7*(i*9+j*3+k)+1)
                  moments(7*(i*9+j*3+k)+4)=-moments(7*(i*9+j*3+k)+4)
               end if
            end do
         end do
      end do
   end subroutine reflect_moments_x
   subroutine reflect_moments_y(moments)
      implicit none
      real(WP), dimension(0:), intent(inout) :: moments !< Needs to be of size (0:188)
      integer :: i,j,k,n
      real(WP) :: temp
      do k=0,2
         do j=0,2
            do i=0,2
               if (j.eq.0) then
                  do n=0,6
                     if (n.eq.2.or.n.eq.5) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=-moments(7*(i*9+2*3+k)+n)
                        moments(7*(i*9+2*3+k)+n)=-temp
                     else
                        temp = moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=+moments(7*(i*9+2*3+k)+n)
                        moments(7*(i*9+2*3+k)+n)=+temp
                     end if
                  end do
               else if (j.eq.1) then
                  moments(7*(i*9+j*3+k)+2)=-moments(7*(i*9+j*3+k)+2)
                  moments(7*(i*9+j*3+k)+5)=-moments(7*(i*9+j*3+k)+5)
               end if
            end do
         end do
      end do
   end subroutine reflect_moments_y
   subroutine reflect_moments_z(moments)
      implicit none
      real(WP), dimension(0:), intent(inout) :: moments !< Needs to be of size (0:188)
      integer :: i,j,k,n
      real(WP) :: temp
      do k=0,2
         do j=0,2
            do i=0,2
               if (k.eq.0) then
                  do n=0,6
                     if (n.eq.3.or.n.eq.6) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=-moments(7*(i*9+j*3+2)+n)
                        moments(7*(i*9+j*3+2)+n)=-temp
                     else
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=+moments(7*(i*9+j*3+2)+n)
                        moments(7*(i*9+j*3+2)+n)=+temp
                     end if
                  end do
               else if (k.eq.1) then
                  moments(7*(i*9+j*3+k)+3)=-moments(7*(i*9+j*3+k)+3)
                  moments(7*(i*9+j*3+k)+6)=-moments(7*(i*9+j*3+k)+6)
               end if
            end do
         end do
      end do
   end subroutine reflect_moments_z
   subroutine reflect_moments_xy(moments)
      implicit none
      real(WP), dimension(0:), intent(inout) :: moments !< Needs to be of size (0:188)
      integer :: i,j,k,n
      real(WP) :: temp
      do k=0,2
         do j=0,2
            do i=0,2
               if (i.eq.j) then
                  do n=0,6
                     if (n.eq.1.or.n.eq.4) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(i*9+j*3+k)+n+1)
                        moments(7*(i*9+j*3+k)+n+1)=temp
                     end if
                  end do
               else if (i.gt.j) then
                  do n=0,6
                     if (n.eq.1.or.n.eq.4) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(j*9+i*3+k)+n+1)
                        moments(7*(j*9+i*3+k)+n+1)=temp
                        temp = moments(7*(j*9+i*3+k)+n)
                        moments(7*(j*9+i*3+k)+n)=moments(7*(i*9+j*3+k)+n+1)
                        moments(7*(i*9+j*3+k)+n+1)=temp
                     else if (n.eq.0.or.n.eq.3.or.n.eq.6) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(j*9+i*3+k)+n)
                        moments(7*(j*9+i*3+k)+n)=temp
                     end if
                  end do
               end if
            end do
         end do
      end do
   end subroutine reflect_moments_xy
   subroutine reflect_moments_yz(moments)
      implicit none
      real(WP), dimension(0:), intent(inout) :: moments !< Needs to be of size (0:188)
      integer :: i,j,k,n
      real(WP) :: temp
      do k=0,2
         do j=0,2
            do i=0,2
               if (j.eq.k) then
                  do n=0,6
                     if (n.eq.2.or.n.eq.5) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(i*9+j*3+k)+n+1)
                        moments(7*(i*9+j*3+k)+n+1)=temp
                     end if
                  end do
               else if (j.gt.k) then
                  do n=0,6
                     if (n.eq.2.or.n.eq.5) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(i*9+k*3+j)+n+1)
                        moments(7*(i*9+k*3+j)+n+1)=temp
                        temp = moments(7*(i*9+k*3+j)+n)
                        moments(7*(i*9+k*3+j)+n)=moments(7*(i*9+j*3+k)+n+1)
                        moments(7*(i*9+j*3+k)+n+1)=temp
                     else if (n.eq.0.or.n.eq.1.or.n.eq.4) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(i*9+k*3+j)+n)
                        moments(7*(i*9+k*3+j)+n)=temp
                     end if
                  end do
               end if
            end do
         end do
      end do
   end subroutine reflect_moments_yz
   subroutine reflect_moments_xz(moments)
      implicit none
      real(WP), dimension(0:), intent(inout) :: moments !< Needs to be of size (0:188)
      integer :: i,j,k,n
      real(WP) :: temp
      do k=0,2
         do j=0,2
            do i=0,2
               if (i.eq.k) then
                  do n=0,6
                     if (n.eq.1.or.n.eq.4) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(i*9+j*3+k)+n+2)
                        moments(7*(i*9+j*3+k)+n+2)=temp
                     end if
                  end do
               else if (i.gt.k) then
                  do n=0,6
                     if (n.eq.1.or.n.eq.4) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(k*9+j*3+i)+n+2)
                        moments(7*(k*9+j*3+i)+n+2)=temp
                        temp = moments(7*(k*9+j*3+i)+n)
                        moments(7*(k*9+j*3+i)+n)=moments(7*(i*9+j*3+k)+n+2)
                        moments(7*(i*9+j*3+k)+n+2)=temp
                     else if (n.eq.0.or.n.eq.2.or.n.eq.5) then
                        temp=moments(7*(i*9+j*3+k)+n)
                        moments(7*(i*9+j*3+k)+n)=moments(7*(k*9+j*3+i)+n)
                        moments(7*(k*9+j*3+i)+n)=temp
                     end if
                  end do
               end if
            end do
         end do
      end do
   end subroutine reflect_moments_xz
"""

print(reflect_subroutines, file=file, end="")
print("end module r2pnet", file=file)

file.close()