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

double precision function disp_DnX(l_grad)
   use common_arrays_C, only: nat, dxyz, cell_ijk, vab
   use molkst_C, only : numat, l123, l1u, l2u, l3u, method_PM6_DH2X, e_disp
   use iso_c_binding, only: c_int, c_double
#ifdef GPU
   use mod_vars_cuda, only: lgpu
#endif
   implicit none
   logical, intent (in) :: l_grad
!
!  Local variables
!
   integer :: i, j, k, l, iii, jjj, kkkk, i_cell, j_cell, iX(100), jOorN(100)
   double precision :: Rab, sum, sum2, sum3,  a_X(3,3), b_X(3,3)
   logical :: first = .true.
   logical, external :: connected
   double precision, external :: distance
#ifdef GPU
   logical :: try_gpu
   integer :: npairs, idx, istat_gpu, istat_env
   character(len=32) :: env_disp
   real(c_double), allocatable :: sum2_list(:), sum3_list(:), rab_list(:), val_list(:), deriv_list(:)
   interface
     integer(c_int) function mopac_cuda_disp_eval(npairs, sum2, sum3, rab, val_out, deriv_out) bind(C, name="mopac_cuda_disp_eval")
       use iso_c_binding
       integer(c_int), value :: npairs
       real(c_double) :: sum2(*), sum3(*), rab(*), val_out(*), deriv_out(*)
     end function mopac_cuda_disp_eval
   end interface
#endif
   save
      if (first) then
          first = .false.
          if (method_PM6_DH2X) then
!
! Add in Rezac and Hobza's correction: "A halogen-bonding correction for the semiempirical PM6 method"
! Chem. Phys. Lett. 506 286-289 (2011)
!
            a_X(1,1) = 1.0489d12  ! Cl - N
            a_X(2,1) = 1.0226d5   ! Br - N
            a_X(3,1) = 1.2751d12  !  I - N
            a_X(1,2) = 4.6783d8   ! Cl - O
            a_X(2,2) = 9.6021d3   ! Br - O
            a_X(3,2) = 6.0912d5   !  I - O
            b_X(1,1) = -9.946d0   ! Cl - N
            b_X(2,1) = -3.236d0   ! Br - N
            b_X(3,1) = -9.534d0   !  I - N
            b_X(1,2) = -6.867d0   ! Cl - O
            b_X(2,2) = -2.900d0   ! Br - O
            b_X(3,2) = -4.154d0   !  I - O
          else
!
! Parameters for the "X" part in the "D3H4X" method
!
! Use Brahmkshatriya, et al.: "Quantum Mechanical Scoring: Structural and Energetic Insights into
! Cyclin-Dependent Kinase 2 Inhibition by Pyrazolo[1,5-a]pyrimidines" Current Computer-Aided Drug Design, 2013, 9, 118-129
! Table 2
            a_X(1,1) = 1.049d12   ! Cl - N
            a_X(2,1) = 5.560d4    ! Br - N
            a_X(3,1) = 5.237d8    !  I - N
            a_X(1,2) = 1.871d9    ! Cl - O
            a_X(2,2) = 2.160d4    ! Br - O
            a_X(3,2) = 2.436d6    !  I - O
            a_X(3,3) = 1.051d6    !  I - S
            b_X(1,1) = -9.95d0    ! Cl - N
            b_X(2,1) = -3.04d0    ! Br - N
            b_X(3,1) = -6.77d0    !  I - N
            b_X(1,2) = -7.44d0    ! Cl - O
            b_X(2,2) = -3.30d0    ! Br - O
            b_X(3,2) = -4.71d0    !  I - O
            b_X(3,3) = -3.82d0    !  I - S
          end if
          iX(17) = 1
          iX(35) = 2
          iX(53) = 3
          jOorN(7)  = 1
          jOorN(8)  = 2
          jOorN(16) = 3
      end if
#ifdef GPU
      try_gpu = lgpu
      env_disp = ''
      istat_env = 1
      call get_environment_variable('MOPAC_DISP_GPU', env_disp, status=istat_env)
      if (istat_env == 0) then
        env_disp = adjustl(env_disp)
        if (len_trim(env_disp) > 0) then
          select case (env_disp(1:1))
          case ('0','f','F','n','N','o','O')
            try_gpu = .false.
          case default
            try_gpu = .true.
          end select
        end if
      end if
      if (.not. lgpu) try_gpu = .false.
      if (try_gpu .and. .not. l_grad) then
        npairs = 0
        do i = 1, numat
          if (nat(i) /= 17 .and. nat(i) /= 35 .and. nat(i) /= 53) cycle
          k = nat(i)
          do j = 1, numat
            select case (nat(j))
            case (7, 8, 16)
              if (k /= 53 .and. nat(j) == 16) cycle
              npairs = npairs + 1
            end select
          end do
        end do
        if (npairs > 0) then
          allocate(sum2_list(npairs), sum3_list(npairs), rab_list(npairs), &
                   val_list(npairs), deriv_list(npairs))
          idx = 0
          do i = 1, numat
            if (nat(i) /= 17 .and. nat(i) /= 35 .and. nat(i) /= 53) cycle
            k = nat(i)
            do j = 1, numat
              select case (nat(j))
              case (7, 8, 16)
                if (k /= 53 .and. nat(j) == 16) cycle
                l = nat(j)
                Rab = distance(i, j)
                sum2 = a_X(iX(k),jOorN(l))
                sum3 = b_X(iX(k),jOorN(l))
                idx = idx + 1
                sum2_list(idx) = sum2
                sum3_list(idx) = sum3
                rab_list(idx) = Rab
              end select
            end do
          end do
         istat_gpu = mopac_cuda_disp_eval(int(npairs, c_int), sum2_list, sum3_list, rab_list, val_list, deriv_list)
         if (istat_gpu == 0) then
            sum = 0.d0
            do idx = 1, npairs
              sum = sum + val_list(idx)
            end do
            e_disp = e_disp + sum
            deallocate(sum2_list, sum3_list, rab_list, val_list, deriv_list)
            disp_DnX = sum
            return
          end if
          deallocate(sum2_list, sum3_list, rab_list, val_list, deriv_list)
        end if
      end if
#endif
      sum = 0.d0
      do i = 1, numat
        if (nat(i) /= 17 .and. nat(i) /= 35 .and. nat(i) /= 53) cycle ! crude, but fast
        k = nat(i)
        do j = 1, numat
          select case (nat(j))
          case (7, 8, 16)
            if (k /= 53 .and. nat(j) == 16) cycle  ! If sulfur, only select iodine
            l = nat(j)
            Rab = distance(i, j)
            sum2 = a_X(iX(k),jOorN(l))
            sum3 = b_X(iX(k),jOorN(l))
            sum = sum + sum2*exp(sum3*Rab)
            if (l_grad) then
              if (connected(i,j, 8.d0**2)) then
  !
  !   kkkk is the cell that atom j is in, relative to atom i
  !
                iii = l123*(i - 1)
                jjj = l123*(j - 1)
                kkkk = (l3u - cell_ijk(3)) + (2*l3u + 1)*(l2u - cell_ijk(2) + (2*l2u + 1)*(l1u - cell_ijk(1)))
                i_cell = iii + kkkk
                j_cell = jjj - kkkk
                do l = 1,3
                  dxyz(i_cell*3 + l) = dxyz(i_cell*3 + l) + Vab(l)*sum2*sum3*exp(sum3*Rab)/Rab
                  dxyz(j_cell*3 + l) = dxyz(j_cell*3 + l) - Vab(l)*sum2*sum3*exp(sum3*Rab)/Rab
                end do
              end if
            end if
          end select
        end do
      end do
      disp_DnX = sum
      e_disp = e_disp + sum
      return
  end function disp_DnX
  subroutine print_post_scf_corrections
  use molkst_C, only : keywrd, E_disp, E_hb, P_Hbonds
  use common_arrays_C, only: H_energy, H_txt
  use chanel_C, only : iw
  implicit none
  double precision :: sum, sum1
  double precision, external :: reada
  integer :: i, j, k
    if (index(keywrd," DISP(") > 0) then
      write(iw,'(/47x,a)')" List of hydrogen bonds found"
      write(iw,'(3x,a,12x,a,16x,a,11x,a,23x,a,17x,a)')"No.", "Donor", &
      "R(D-H)", "Hydrogen",  "Acceptor", "H-bond energy"
      sum1 = -abs(reada(keywrd, index(keywrd," DISP(") + 5))
      k = 0
      do
        sum = 0.d0
        j = 0
        do i = 1, P_hbonds
          if (sum > H_energy(i)) then
            sum = H_energy(i)
            j = i
          end if
        end do
        if (sum > sum1) exit
        k = k + 1
        write(iw,'(i5,3x,a)')k, trim(H_txt(j))
        H_energy(j) = 10.d0
      end do
    end if
    if (index(keywrd, "0SCF") /= 0) then
      write(iw,'(/10x,"DISPERSION ENERGY       =", f17.5, a)') e_disp, " KCAL/MOL"
      write(iw,'(10x,"H-BOND ENERGY           =", f17.5, a,/)') e_hb, " KCAL/MOL"
    end if
  end subroutine print_post_scf_corrections
