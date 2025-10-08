module gpu_scf_stream_interfaces
  use iso_c_binding
  implicit none
  interface
    logical(c_bool) function mopac_cuda_scf_stream_supported() bind(C)
      import :: c_bool
    end function mopac_cuda_scf_stream_supported

    subroutine mopac_cuda_scf_stream_register(cookie) bind(C)
      import :: c_ptr
      type(c_ptr), value :: cookie
    end subroutine mopac_cuda_scf_stream_register

    subroutine mopac_cuda_scf_stream_publish(cookie, ia, ib, ja, jb, len, wj, wk, status) bind(C)
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: cookie
      integer(c_int), value :: ia, ib, ja, jb, len
      real(c_double), intent(in) :: wj(len), wk(len)
      integer(c_int), intent(out) :: status
    end subroutine mopac_cuda_scf_stream_publish

    subroutine mopac_cuda_scf_stream_finalize(cookie, status) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), value :: cookie
      integer(c_int), intent(out) :: status
    end subroutine mopac_cuda_scf_stream_finalize
  end interface
end module gpu_scf_stream_interfaces
