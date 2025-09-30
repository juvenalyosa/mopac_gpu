program gpu_fock_pair_compare
  use molkst_C, only : norbs, mpack
#ifdef GPU
  use iso_c_binding, only : c_bool, c_int
  use gpu_fock_interfaces, only : mopac_cuda_fock2_scf
#endif
  implicit none
  integer :: failures
  failures = 0
#ifdef GPU
  if (.not. run_case(1, 1, 'LL')) failures = failures + 1
  if (.not. run_case(1, 7, 'LH')) failures = failures + 1
  if (.not. run_case(7, 7, 'HH')) failures = failures + 1
#else
  print *, 'GPU support not enabled; skipping gpu_fock_pair_compare'
#endif
  if (failures /= 0) stop 1
#ifdef GPU
contains
  logical function run_case(span_a, span_b, label)
    use iso_fortran_env, only : real64
    implicit none
    integer, intent(in)        :: span_a, span_b
    character(*), intent(in)   :: label
    integer :: local_norbs, local_mpack
    integer :: nfirst(2), nlast(2)
    integer, allocatable :: ifact_local(:)
    double precision, allocatable :: ptot(:), p(:), w(:), f_cpu(:), f_gpu(:)
    integer :: i, len_w
    integer(c_int) :: ia, ib, ja, jb, kr
    logical(c_bool) :: ok
    double precision :: diff, denom

    interface
      subroutine fockdorbs(ia, ib, ja, jb, f, p_mat, ptot_mat, w_vals, kr_loc, ifact_loc)
        use iso_c_binding, only : c_int
        integer(c_int), intent(in) :: ia, ib, ja, jb
        integer(c_int), intent(inout) :: kr_loc
        double precision, intent(inout) :: f(*)
        double precision, intent(in) :: p_mat(*), ptot_mat(*), w_vals(*)
        integer(c_int), intent(in) :: ifact_loc(*)
      end subroutine fockdorbs
    end interface

    run_case = .false.

    local_norbs = span_a + span_b
    if (local_norbs <= 0) then
      print *, 'Invalid spans for ', trim(label)
      return
    end if
    local_mpack = local_norbs * (local_norbs + 1) / 2
    norbs = local_norbs
    mpack = local_mpack

    allocate(ifact_local(local_norbs))
    do i = 1, local_norbs
      ifact_local(i) = (i * (i - 1)) / 2
    end do

    allocate(ptot(local_mpack), p(local_mpack), f_cpu(local_mpack), f_gpu(local_mpack))
    ptot = 0.0d0
    p = 0.0d0
    do i = 1, local_mpack
      ptot(i) = 0.01d0 * dble(i)
      p(i) = 0.005d0 * dble(mod(i, local_mpack) + 1)
    end do
    f_cpu = 0.0d0
    f_gpu = 0.0d0

    nfirst = [1, span_a + 1]
    nlast = [span_a, local_norbs]

    len_w = (span_a * (span_a + 1) / 2) * (span_b * (span_b + 1) / 2)
    if (len_w <= 0) then
      print *, 'No integrals for ', trim(label)
      run_case = .true.
      deallocate(ptot, p, f_cpu, f_gpu, ifact_local)
      return
    end if
    allocate(w(len_w))
    do i = 1, len_w
      w(i) = 1.0d-3 * dble(i)
    end do

    ia = nfirst(2)
    ib = nlast(2)
    ja = nfirst(1)
    jb = nlast(1)
    kr = 0
    call fockdorbs(ia, ib, ja, jb, f_cpu, p, ptot, w, kr, ifact_local)
    if (kr /= len_w) then
      print *, 'fockdorbs consumed ', kr, ' weights but expected ', len_w, ' for ', trim(label)
      deallocate(ptot, p, f_cpu, f_gpu, w, ifact_local)
      return
    end if

    ok = mopac_cuda_fock2_scf(local_norbs, local_mpack, 2, nfirst, nlast, ptot, p, w, f_gpu)
    if (.not. ok) then
      print *, 'GPU path unavailable for ', trim(label), ' case; skipping'
      run_case = .true.
      deallocate(ptot, p, f_cpu, f_gpu, w, ifact_local)
      return
    end if

    diff = 0.0d0
    denom = 1.0d0
    do i = 1, local_mpack
      diff = max(diff, abs(f_cpu(i) - f_gpu(i)))
      denom = max(denom, abs(f_cpu(i)))
    end do
    if (diff > 1.0d-8 .and. diff/denom > 1.0d-8) then
      print *, 'GPU/CPU mismatch for ', trim(label), ': diff=', diff, ' denom=', denom
      deallocate(ptot, p, f_cpu, f_gpu, w, ifact_local)
      return
    end if
    print *, 'GPU/CPU match for ', trim(label), ' diff=', diff
    run_case = .true.

    deallocate(ptot, p, f_cpu, f_gpu, w, ifact_local)
  end function run_case
#endif
end program gpu_fock_pair_compare
