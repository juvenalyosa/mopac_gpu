module gpu_scf_stream_driver
  use iso_c_binding
  use gpu_scf_stream_interfaces, only: mopac_cuda_scf_stream_supported, &
                                       mopac_cuda_scf_stream_register,  &
                                       mopac_cuda_scf_stream_publish,   &
                                       mopac_cuda_scf_stream_finalize
  implicit none
  private

  type, bind(C) :: gpu_scf_stream_cookie
     integer(c_int) :: norbs          = 0_c_int
     integer(c_int) :: mpack          = 0_c_int
     integer(c_int) :: numat          = 0_c_int
     integer(c_int) :: periodic_flag  = 0_c_int
     integer(c_int) :: has_exchange   = 0_c_int
     type(c_ptr)    :: ptot           = c_null_ptr
     type(c_ptr)    :: p              = c_null_ptr
     type(c_ptr)    :: f              = c_null_ptr
     type(c_ptr)    :: nfirst         = c_null_ptr
     type(c_ptr)    :: nlast          = c_null_ptr
  end type gpu_scf_stream_cookie

  public :: gpu_scf_stream_fock

contains

  logical function gpu_scf_stream_fock(norbs, mpack, numat, nfirst, nlast, ptot, p, w, wj, wk, periodic_flag, f)
    use iso_c_binding, only: c_int, c_double, c_ptr, c_loc
    implicit none
    integer, intent(in) :: norbs, mpack, numat
    integer, intent(in), target :: nfirst(*), nlast(*)
    double precision, intent(in), target :: ptot(mpack)
    double precision, intent(in), target :: p(mpack)
    double precision, intent(in) :: w(:)
    double precision, intent(in) :: wj(:)
    double precision, intent(in) :: wk(:)
    integer, intent(in) :: periodic_flag
    double precision, intent(inout), target :: f(mpack)

    type(gpu_scf_stream_cookie) :: cookie
    type(c_ptr) :: cookie_ptr
    integer :: ii, jj
    integer :: ia, ib, ja, jb
    integer :: span_i, span_j
    integer :: pairs_i, pairs_j
    integer :: len_block
    integer :: kk
    integer(c_int) :: status
    integer(c_int) :: final_status
    real(c_double), pointer :: block_j(:)
    real(c_double), pointer :: block_k(:)
    integer :: wj_size
    integer :: wk_size
    logical :: supported
    integer :: num_atoms

    gpu_scf_stream_fock = .false.
    supported = (mopac_cuda_scf_stream_supported() .eqv. .true._c_bool)
    if (.not. supported) return

    num_atoms = numat
    if (num_atoms <= 1) return

    cookie%norbs         = int(norbs, kind=c_int)
    cookie%mpack         = int(mpack, kind=c_int)
    cookie%numat         = int(num_atoms, kind=c_int)
    cookie%periodic_flag = int(periodic_flag, kind=c_int)
    if (size(wk) > 0) then
      cookie%has_exchange = 1_c_int
    else
      cookie%has_exchange = 0_c_int
    end if
    cookie%ptot   = c_loc(ptot(1))
    cookie%p      = c_loc(p(1))
    cookie%f      = c_loc(f(1))
    cookie%nfirst = c_loc(nfirst(1))
    cookie%nlast  = c_loc(nlast(1))

    cookie_ptr = c_loc(cookie)
    call mopac_cuda_scf_stream_register(cookie_ptr)

    kk = 0
    status = 0_c_int
    final_status = 0_c_int
    wj_size = size(wj)
    wk_size = size(wk)

    outer_atoms: do ii = 1, num_atoms
      ia = nfirst(ii)
      ib = nlast(ii)
      span_i = ib - ia + 1
      if (span_i <= 0) cycle
      pairs_i = span_i * (span_i + 1) / 2
      if (pairs_i <= 0) cycle

      do jj = 1, ii - 1
        ja = nfirst(jj)
        jb = nlast(jj)
        span_j = jb - ja + 1
        if (span_j <= 0) cycle
        pairs_j = span_j * (span_j + 1) / 2
        if (pairs_j <= 0) cycle

        if (periodic_flag /= 0) then
          len_block = pairs_i * pairs_j
        else
          if (span_i >= 7 .or. span_j >= 7) then
            len_block = pairs_i * pairs_j
          else if (span_i >= 4 .and. span_j >= 4) then
            len_block = 100
          else if (span_i >= 4 .and. span_j == 1) then
            len_block = 10
          else if (span_j >= 4 .and. span_i == 1) then
            len_block = 10
          else if (span_i == 1 .and. span_j == 1) then
            len_block = 1
          else
            len_block = pairs_i * pairs_j
          end if
        end if

        if (len_block <= 0) cycle
        if (kk + len_block > wj_size) then
          status = -3_c_int
          exit outer_atoms
        end if

        block_j => wj(kk + 1 : kk + len_block)
        if (wk_size >= kk + len_block) then
          block_k => wk(kk + 1 : kk + len_block)
        else
          block_k => wj(kk + 1 : kk + len_block)
        end if

        call mopac_cuda_scf_stream_publish(cookie_ptr,                                            &
             int(ia, kind=c_int), int(ib, kind=c_int),                                            &
             int(ja, kind=c_int), int(jb, kind=c_int),                                            &
             int(len_block, kind=c_int), block_j, block_k, status)

        if (status /= 0_c_int) then
          exit outer_atoms
        end if
        kk = kk + len_block
      end do
    end do outer_atoms

    call mopac_cuda_scf_stream_finalize(cookie_ptr, final_status)
    if (status /= 0_c_int .and. final_status == 0_c_int) final_status = status
    if (final_status == 0_c_int) gpu_scf_stream_fock = .true.
  end function gpu_scf_stream_fock

end module gpu_scf_stream_driver
