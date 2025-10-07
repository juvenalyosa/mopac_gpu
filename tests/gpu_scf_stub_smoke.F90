program gpu_scf_stub_smoke
  use iso_c_binding
  use gpu_scf_types,        only : gpu_scf_context, gpu_scf_context_clear, &
                                   GPU_SCF_FLAG_RHF, GPU_SCF_FLAG_USE_DIIS
  use gpu_scf_interfaces,   only : gpu_scf_run, gpu_scf_last_error
  implicit none

  type(gpu_scf_context) :: ctx
  logical :: ok
  character(len=256) :: message

  call gpu_scf_context_clear(ctx)
  ctx%norbs      = 3
  ctx%nalpha     = 2
  ctx%nbeta      = 1
  ctx%mpack      = 6
  ctx%numat      = 1
  ctx%n2elec     = 1
  ctx%max_iter   = 25
  ctx%energy_tol = 1.0d-8
  ctx%density_tol = 1.0d-8
  ctx%flags      = GPU_SCF_FLAG_RHF + GPU_SCF_FLAG_USE_DIIS

  ok = gpu_scf_run(ctx)
  if (ok) then
     write(*,'(1x,a)') 'gpu_scf_run unexpectedly succeeded in stub mode'
     stop 1
  end if

  call gpu_scf_last_error(message)
  if (index(message, 'stub') == 0) then
     write(*,'(1x,a)') 'gpu_scf_last_error did not report stub state:'
     write(*,'(1x,a)') trim(message)
     stop 2
  end if

  write(*,'(1x,a)') 'GPU SCF stub smoke test completed: '//trim(message)
end program gpu_scf_stub_smoke
