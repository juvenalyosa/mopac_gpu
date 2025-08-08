
! Molecular Orbital PACkage (MOPAC)
! Copyright 2021 Virginia Polytechnic Institute and State University
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!    http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

Subroutine eigenvectors_CUSOLVER(eigenvecs, xmat, eigvals, ndim)
  USE chanel_C, only : iw
  use iso_c_binding
  use cusolverDn
  implicit none
  Integer :: ndim
  double precision :: eigenvecs(ndim,ndim), &
                & eigvals(ndim),xmat((ndim*(ndim+1))/2)
  double precision, allocatable, device :: d_a(:,:), d_w(:)
  integer(c_int) :: lwork, info
  double precision, allocatable :: work(:)
  integer(c_int) :: devInfo
  type(cusolverDnHandle) :: handle
  integer(c_int) :: status
  
  ! Start cuSOLVER
  status = cusolverDnCreate(handle)
  
  ! Allocate memory on the GPU
  allocate (d_a(ndim, ndim), d_w(ndim))

  ! Copy matrix to GPU
  call dtpttr( 'u', ndim, xmat, eigenvecs, ndim, info )
  if (info /= 0) stop 'error in dtpttr'
  
  status = cudaMemcpy(d_a, eigenvecs, size(eigenvecs), cudaMemcpyHostToDevice)

  ! Query workspace size
  status = cusolverDnDsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_FILL_MODE_UPPER, ndim, d_a, ndim, d_w, lwork)
  allocate(work(lwork))
  
  ! Perform diagonalization
  status = cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_FILL_MODE_UPPER, ndim, d_a, ndim, d_w, work, lwork, devInfo)
  
  ! Copy results back to CPU
  status = cudaMemcpy(eigenvecs, d_a, size(eigenvecs), cudaMemcpyDeviceToHost)
  status = cudaMemcpy(eigvals, d_w, size(eigvals), cudaMemcpyDeviceToHost)
  
  ! Clean up
  deallocate(work)
  deallocate(d_a, d_w)
  status = cusolverDnDestroy(handle)
  
  if (devInfo /= 0)  write(iw,*) ' cusolverDnDsyevd Diagonalization error., CODE =',devInfo
  
End Subroutine eigenvectors_CUSOLVER
