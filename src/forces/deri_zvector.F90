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

      subroutine deri_zvector(minear, ninear, scalar, diag, zvec, cvec, aorel, ok)
!-----------------------------------------------
!   M o d u l e s
!-----------------------------------------------
      use molkst_C, only : norbs, nopen, fract, numat, mpack, keywrd
      use meci_C, only : nelec, nmos, nbo, lab, nstate, vectci, dijkl
      use common_arrays_C, only : c, w, eigs, nfirst, nlast
      use chanel_C, only : iw
#ifdef GPU
      use gpu_conv_fock, only : gpu_conv_w_ensure_host
#endif
      implicit none
      integer, intent(in) :: minear, ninear
      double precision, intent(in) :: scalar(minear), diag(minear)
      double precision, intent(out) :: zvec(minear), cvec(ninear), aorel(mpack)
      logical, intent(out) :: ok
!***********************************************************************
!
!   Z-VECTOR FORM OF THE RELAXATION PART OF THE ANALYTICAL C.I. GRADIENT
!
!   DERI2 solves, for every Cartesian coordinate v, the linear system
!   M y_v = f_v  (M: the scaled electronic Hessian applied by DERI22, f_v:
!   the scaled non-relaxed Fock derivative from DERI1), then evaluates the
!   relaxation of the C.I. energy from x_v = S y_v (S = diag(scalar)) and
!   the within-block Fock derivative fd_v (DERI23, DIJKL2, MECID, MECIH).
!   Every step after the solve is linear, so the relaxation contribution is
!
!       R_v = g . x_v + c . fd_v = z . f_v + c . fd_v,   M z = S g
!
!   (M is symmetric).  This routine builds g and c from the C.I. vectors
!   and the M.O. integrals once, and solves the single system for z, so
!   the 3*NUMAT relaxation solves of DERI2 become one.  DERNVO then adds
!   (z . f_v + c . fd_v) * fpc_9 / nstate to each gradient component.
!
!   Both dot products are traces over the A.O. Fock derivative F_v (DERI1:
!   f_v = scalar * (C' F_v C) over the off-diagonal blocks, fd_v = C' F_v C
!   over the C.I.-active diagonal blocks), so they are returned as one packed
!   A.O. matrix AOREL with  z . f_v + c . fd_v = sum(F_v * AOREL).
!
!   OK is false when the solve fails; DERNVO then uses DERI2.
!
!***********************************************************************
      double precision, allocatable :: wmo(:,:), emat(:,:), q(:,:), t(:,:), &
        dp(:), foc2(:), xyp(:,:,:,:), eiga(:), cidiag(:), cimat(:), civec(:), &
        gamma(:), gvec(:), rhs(:), r(:), p(:), ap(:), bscr(:), work(:,:), &
        foc2w(:), fci(:)
      double precision :: r0, val, scal, diffe, k_l, alpha, beta, rr, rr_new, pap, &
        rmax, tol, const
      double precision :: wj(1) = 0.d0, wk(1) = 0.d0
      integer :: i, j, k, l, ll, ij, kl, nend, loop, ninit, n1, n2, ncol, iter, &
        maxit
      logical :: lij, lkl, debug
      double precision, external :: ddot
!
      ok = .false.
      debug = index(keywrd, ' DERI2') /= 0 .or. index(keywrd, 'DERNVO') /= 0
#ifdef GPU
      call gpu_conv_w_ensure_host()
#endif
      allocate (wmo(norbs, nmos), emat(norbs, norbs), q(norbs, norbs), t(norbs, norbs), &
        dp(mpack), foc2(mpack), xyp(nmos, nmos, nmos, nmos), eiga(nmos), cidiag(lab), &
        cimat((lab*(lab + 1))/2), civec(lab), gamma(nmos), gvec(minear), rhs(minear), &
        r(minear), p(minear), ap(minear), bscr(minear), work(norbs, norbs), &
        foc2w(norbs**2), fci(ninear))
!
!  1.  C.I. ENERGY DERIVATIVE WITH RESPECT TO THE ACTIVE FOCK EIGENVALUES
!      (GAMMA) AND TO THE ACTIVE INTEGRALS <IJ|KL> (ONE PROBE PER SET OF
!      EIGHT SYMMETRY-EQUIVALENT ELEMENTS, AS DIJKL2 WRITES THEM), AND THE
!      ADJOINT OF DIJKL2: WMO(M,A) = D(C.I. ENERGY) / D(DC(M,A)).
!
      eiga = 0.d0
      xyp = 0.d0
      r0 = ci_form(eiga, xyp)
      do i = 1, nmos
        eiga(i) = 1.d0
        gamma(i) = ci_form(eiga, xyp) - r0
        eiga(i) = 0.d0
      end do
      wmo = 0.d0
      ij = 0
      do i = 1, nmos
        do j = 1, i
          ij = ij + 1
          lij = i == j
          kl = 0
          do k = 1, i
            if (k == i) then
              ll = j
            else
              ll = k
            end if
            do l = 1, ll
              kl = kl + 1
              lkl = k == l
              call set_xy(i, j, k, l, 1.d0)
              val = ci_form(eiga, xyp) - r0
              call set_xy(i, j, k, l, 0.d0)
              if (val == 0.d0) cycle
              if (lij .and. lkl .and. j == k) then
                wmo(:, i) = wmo(:, i) + 4.d0*val*dijkl(:, j, kl)
              else
                if (lij) then
                  wmo(:, i) = wmo(:, i) + 2.d0*val*dijkl(:, j, kl)
                else
                  wmo(:, i) = wmo(:, i) + val*dijkl(:, j, kl)
                  wmo(:, j) = wmo(:, j) + val*dijkl(:, i, kl)
                end if
                if (lkl) then
                  wmo(:, k) = wmo(:, k) + 2.d0*val*dijkl(:, l, ij)
                else
                  wmo(:, k) = wmo(:, k) + val*dijkl(:, l, ij)
                  wmo(:, l) = wmo(:, l) + val*dijkl(:, k, ij)
                end if
              end if
            end do
          end do
        end do
      end do
!
!  2.  WALK THE LOOPS OF DERI23: COEFFICIENTS OF FD (CVEC) AND OF THE
!      DIAGONAL-BLOCK FOCK RESPONSE (MATRIX EMAT, SEE DERI22: FCI(L) =
!      (C' G C)(I,J) FOR PAIRS, -(C' G C)(A,A) FOR THE EIGENVALUES), AND
!      THE DIRECT COEFFICIENTS OF X (GVEC).
!
      cvec = 0.d0
      emat = 0.d0
      l = 1
      nend = 0
      n2 = 0
      ninit = 1
      do loop = 1, 3
        ninit = nend + 1
        nend = nend + nbo(loop)
        n1 = max(ninit, nelec + 1)
        n2 = min(nend, nelec + nmos)
        if (n2 < n1) cycle
        do i = n1, n2
          if (i > ninit) then
            do j = ninit, i - 1
              diffe = eigs(i) - eigs(j)
              if (abs(diffe) > 1.d-4) then
                k_l = (wc(j, i) - wc(i, j))/diffe
                cvec(l) = k_l
                emat(i, j) = emat(i, j) - 0.5d0*k_l
                emat(j, i) = emat(j, i) - 0.5d0*k_l
              end if
              l = l + 1
            end do
          end if
        end do
      end do
      ncol = n2 - ninit + 1
      if (ncol > 0 .and. n2 < norbs) then
        do j = ninit, n2
          do i = n2 + 1, norbs
            diffe = eigs(i) - eigs(j)
            if (abs(diffe) > 1.d-4) then
              k_l = (wc(j, i) - wc(i, j))/diffe
              cvec(l) = k_l
              emat(i, j) = emat(i, j) - 0.5d0*k_l
              emat(j, i) = emat(j, i) - 0.5d0*k_l
            end if
            l = l + 1
          end do
        end do
      end if
      do i = 1, nmos
        emat(nelec + i, nelec + i) = emat(nelec + i, nelec + i) - gamma(i)
      end do
!
!  Q = C' G(C EMAT C') C, with G the two-electron operator of DERI22.
!
      call mxm (c, norbs, emat, norbs, t, norbs)
      call mxmt (t, norbs, c, norbs, q, norbs)
      l = 0
      do i = 1, norbs
        do j = 1, i
          l = l + 1
          dp(l) = q(i, j)
        end do
      end do
      foc2 = 0.d0
      call fock2(foc2, dp, 0.5d0*dp, w, wj, wk, numat, nfirst, nlast, 2)
      l = 0
      do i = 1, norbs
        do j = 1, i
          l = l + 1
          t(i, j) = foc2(l)
          t(j, i) = foc2(l)
        end do
      end do
      call mxm (t, norbs, c, norbs, q, norbs)
      call mtxm (c, norbs, q, norbs, t, norbs)
!
!  G = A + 2 Q(I,J) OVER THE OFF-DIAGONAL BLOCKS, ORDERED AS IN DERI1/DERI23.
!
      const = 1.d-3
      l = 1
      if (nbo(2) > 0 .and. nbo(1) > 0) then
        scal = 1.d0/(2.d0 - fract + const)
        do j = 1, nbo(1)
          do i = nbo(1) + 1, nopen
            gvec(l) = scal*(wc(j, i) - wc(i, j)) + 2.d0*t(i, j)
            l = l + 1
          end do
        end do
      end if
      if (nbo(3) > 0 .and. nbo(1) > 0) then
        scal = 0.5d0
        do j = 1, nbo(1)
          do i = nopen + 1, norbs
            gvec(l) = scal*(wc(j, i) - wc(i, j)) + 2.d0*t(i, j)
            l = l + 1
          end do
        end do
      end if
      if (nbo(3) /= 0 .and. nbo(2) /= 0) then
        scal = 1.d0/(fract + const)
        do j = nbo(1) + 1, nopen
          do i = nopen + 1, norbs
            gvec(l) = scal*(wc(j, i) - wc(i, j)) + 2.d0*t(i, j)
            l = l + 1
          end do
        end do
      end if
!
!  3.  SOLVE M Z = S G BY CONJUGATE GRADIENTS (M = DERI22, SYMMETRIC).
!
      rhs = scalar*gvec
      zvec = 0.d0
      r = rhs
      p = r
      rr = ddot(minear, r, 1, r, 1)
      rmax = maxval(abs(rhs))
      tol = max(1.d-10*rmax, 1.d-14)
      maxit = 500
      if (rmax == 0.d0) then
        ok = .true.
        iter = 0
      else
        do iter = 1, maxit
          bscr = p
          call deri22(c, bscr, work, foc2w, ap, minear, fci, w, diag, scalar, ninear)
          pap = ddot(minear, p, 1, ap, 1)
          if (pap <= 0.d0) exit
          alpha = rr/pap
          zvec = zvec + alpha*p
          r = r - alpha*ap
          if (maxval(abs(r)) <= tol) then
            ok = .true.
            exit
          end if
          rr_new = ddot(minear, r, 1, r, 1)
          beta = rr_new/rr
          rr = rr_new
          p = r + beta*p
        end do
      end if
!
!  4.  AOREL = C X C' PACKED FOR A SYMMETRIC F (OFF-DIAGONAL ELEMENTS
!      COUNTED TWICE), X(I,J) = Z*SCALAR AND C OVER THE (I,J) OF DERI1.
!
      if (ok) then
        emat = 0.d0
        l = 1
        if (nbo(2) > 0 .and. nbo(1) > 0) then
          do j = 1, nbo(1)
            do i = nbo(1) + 1, nopen
              emat(i, j) = zvec(l)*scalar(l)
              l = l + 1
            end do
          end do
        end if
        if (nbo(3) > 0 .and. nbo(1) > 0) then
          do j = 1, nbo(1)
            do i = nopen + 1, norbs
              emat(i, j) = zvec(l)*scalar(l)
              l = l + 1
            end do
          end do
        end if
        if (nbo(3) /= 0 .and. nbo(2) /= 0) then
          do j = nbo(1) + 1, nopen
            do i = nopen + 1, norbs
              emat(i, j) = zvec(l)*scalar(l)
              l = l + 1
            end do
          end do
        end if
        l = 1
        nend = 0
        do loop = 1, 3
          ninit = nend + 1
          nend = nend + nbo(loop)
          n1 = max(ninit, nelec + 1)
          n2 = min(nend, nelec + nmos)
          if (n2 < n1) cycle
          do i = n1, n2
            do j = ninit, i - 1
              emat(i, j) = emat(i, j) + cvec(l)
              l = l + 1
            end do
          end do
        end do
        ncol = n2 - ninit + 1
        if (ncol > 0 .and. n2 < norbs) then
          do j = ninit, n2
            do i = n2 + 1, norbs
              emat(i, j) = emat(i, j) + cvec(l)
              l = l + 1
            end do
          end do
        end if
        call mxm (c, norbs, emat, norbs, t, norbs)
        call mxmt (t, norbs, c, norbs, q, norbs)
        l = 0
        do i = 1, norbs
          do j = 1, i - 1
            l = l + 1
            aorel(l) = q(i, j) + q(j, i)
          end do
          l = l + 1
          aorel(l) = q(i, i)
        end do
      end if
      if (debug .or. .not. ok) then
        write(iw, '(a,i5,a,es10.3,a,es10.3,a,l2)') ' DERI_ZVECTOR: CG ITERATIONS', &
          min(iter, maxit), '  MAX RESIDUAL', maxval(abs(r)), '  MAX RHS', rmax, '  CONVERGED', ok
      end if
      deallocate (wmo, emat, q, t, dp, foc2, xyp, eiga, cidiag, cimat, civec, gamma, gvec, &
        rhs, r, p, ap, bscr, work, foc2w, fci)
      return
!
      contains
!
!  WMO OVER THE C.I.-ACTIVE COLUMNS, ZERO ELSEWHERE.
!
      double precision function wc(m, col)
        integer, intent(in) :: m, col
        if (col > nelec .and. col <= nelec + nmos) then
          wc = wmo(m, col - nelec)
        else
          wc = 0.d0
        end if
      end function wc
!
      subroutine set_xy(i, j, k, l, v)
        integer, intent(in) :: i, j, k, l
        double precision, intent(in) :: v
        xyp(i, j, k, l) = v
        xyp(i, j, l, k) = v
        xyp(j, i, k, l) = v
        xyp(j, i, l, k) = v
        xyp(k, l, i, j) = v
        xyp(k, l, j, i) = v
        xyp(l, k, i, j) = v
        xyp(l, k, j, i) = v
      end subroutine set_xy
!
!  SUM OVER THE STATES OF V' H V, H THE C.I. MATRIX OF MECID/MECIH FOR THE
!  GIVEN ACTIVE EIGENVALUES AND INTEGRALS (AS IN DERI1 AND DERI2).
!
      double precision function ci_form(emo, xyv)
        double precision, intent(in) :: emo(nmos)
        double precision :: xyv(nmos, nmos, nmos, nmos)
        double precision :: gse, eigw(nmos), s
        integer :: istate, jj
        call mecid(emo, gse, eigw, cidiag, xyv)
        call mecih(cidiag, cimat, nmos, lab, xyv)
        s = 0.d0
        do istate = 1, nstate
          jj = (istate - 1)*lab + 1
          call supdot(civec, cimat, vectci(jj), lab)
          s = s + ddot(lab, vectci(jj), 1, civec, 1)
        end do
        ci_form = s
      end function ci_form
      end subroutine deri_zvector
