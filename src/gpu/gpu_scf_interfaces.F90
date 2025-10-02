module gpu_scf_interfaces
  use iso_c_binding
  use gpu_scf_types, only: gpu_scf_context, GPU_SCF_FLAG_USE_DIIS, &
       GPU_SCF_FLAG_RHF, GPU_SCF_FLAG_UHF, GPU_SCF_FLAG_DEBUG
  implicit none
  private
  public :: gpu_scf_context, gpu_scf_run, gpu_scf_last_error, &
            GPU_SCF_FLAG_USE_DIIS, GPU_SCF_FLAG_RHF, GPU_SCF_FLAG_UHF, GPU_SCF_FLAG_DEBUG

  interface
    function mopac_cuda_scf_run(ctx) bind(C,name='mopac_cuda_scf_run') result(ok)
      import :: c_ptr, c_bool
      type(c_ptr), value :: ctx
      logical(c_bool) :: ok
    end function mopac_cuda_scf_run

    subroutine mopac_cuda_scf_release() bind(C,name='mopac_cuda_scf_release')
    end subroutine mopac_cuda_scf_release

    function mopac_cuda_scf_last_error(buf, len) bind(C,name='mopac_cuda_scf_last_error') result(total)
      import :: c_char, c_size_t
      character(kind=c_char) :: buf(*)
      integer(c_size_t), value :: len
      integer(c_size_t) :: total
    end function mopac_cuda_scf_last_error
  end interface
contains

  logical function gpu_scf_run(ctx) result(success)
    type(gpu_scf_context), intent(inout), target :: ctx
    type(c_ptr) :: ctx_ptr
    logical(c_bool) :: ok
    ctx_ptr = c_loc(ctx)
    ok = mopac_cuda_scf_run(ctx_ptr)
    success = (ok .eqv. .true._c_bool)
  end function gpu_scf_run

  subroutine gpu_scf_last_error(message)
    character(len=*), intent(out) :: message
    integer :: i, limit
    integer(c_size_t) :: total
    integer(c_size_t) :: buf_size
    character(kind=c_char), allocatable :: buf(:)
    character :: template

    if (len(message) <= 0) return

    buf_size = int(len(message) + 1, kind=c_size_t)
    allocate(buf(0:int(buf_size)-1))
    buf = c_null_char

    total = mopac_cuda_scf_last_error(buf, buf_size)
    message = ''
    limit = min(len(message), int(total))
    if (limit <= 0) then
      deallocate(buf)
      return
    end if

    template = ' '
    do i = 1, limit
      if (buf(i-1) == c_null_char) exit
      message(i:i) = transfer(buf(i-1), template)
    end do
    deallocate(buf)
  end subroutine gpu_scf_last_error

end module gpu_scf_interfaces
