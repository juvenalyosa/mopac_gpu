"""ctypes binding of the MOZYME QM/MM session API (include/mopac.h: mozyme_qmmm_init / _step / _finalize).

    from mopac_qmmm_api import QMSession
    qm = QMSession(libpath, atomic_numbers, coords, charge=0, model=0)
    heat, grad_qm, grad_mm = qm.step(coords, mm_coords, mm_charges, ron=0.0, roff=0.0, qm_pot=None)
    qm.close()

Units: Angstroms, electrons, kcal/mol, kcal/(mol Angstrom).  heat is the PM7 heat of formation of the QM
system in the field of the MM charges (it includes the QM-MM electrostatics).  qm_pot (kcal/(mol e), one per
QM atom) is an external potential acting on the Mulliken charges; after a step, qm.mm_pot holds dheat/dq of
each MM charge and qm.qm_charges the Mulliken charges.
"""

import ctypes as C
import numpy as np


class MopacSystem(C.Structure):
    _fields_ = [("natom", C.c_int), ("natom_move", C.c_int), ("charge", C.c_int), ("spin", C.c_int),
                ("model", C.c_int), ("epsilon", C.c_double), ("atom", C.POINTER(C.c_int)),
                ("coord", C.POINTER(C.c_double)), ("nlattice", C.c_int), ("nlattice_move", C.c_int),
                ("pressure", C.c_double), ("lattice", C.POINTER(C.c_double)), ("tolerance", C.c_double),
                ("max_time", C.c_int)]


class MozymeState(C.Structure):
    _fields_ = [("numat", C.c_int), ("nbonds", C.c_void_p), ("ibonds", C.c_void_p), ("iorbs", C.c_void_p),
                ("noccupied", C.c_int), ("ncf", C.c_void_p), ("nvirtual", C.c_int), ("nce", C.c_void_p),
                ("icocc_dim", C.c_int), ("icocc", C.c_void_p), ("icvir_dim", C.c_int), ("icvir", C.c_void_p),
                ("cocc_dim", C.c_int), ("cocc", C.c_void_p), ("cvir_dim", C.c_int), ("cvir", C.c_void_p)]


class MopacProperties(C.Structure):
    _fields_ = [("heat", C.c_double), ("dipole", C.c_double * 3), ("charge", C.c_void_p),
                ("coord_update", C.c_void_p), ("coord_deriv", C.c_void_p), ("freq", C.c_void_p),
                ("disp", C.c_void_p), ("bond_index", C.c_void_p), ("bond_atom", C.c_void_p),
                ("bond_order", C.c_void_p), ("lattice_update", C.c_void_p), ("lattice_deriv", C.c_void_p),
                ("stress", C.c_double * 6), ("nerror", C.c_int), ("error_msg", C.c_void_p)]


class QMSession:
    def __init__(self, libpath, atomic_numbers, coords, charge=0, model=0, spin=0, tolerance=1.0,
                 max_time=86400):
        self.lib = C.CDLL(str(libpath))
        self.n = len(atomic_numbers)
        self._atom = (C.c_int * self.n)(*atomic_numbers)
        self._coord = (C.c_double * (3 * self.n))(*np.asarray(coords, float).ravel())
        self.system = MopacSystem(self.n, self.n, charge, spin, model, 1.0, self._atom, self._coord, 0, 0, 0.0,
                                  None, tolerance, max_time)
        self.state = MozymeState()
        status = C.c_int(1)
        self.lib.mozyme_qmmm_init(C.byref(self.system), C.byref(self.state), C.byref(status))
        if status.value != 0:
            raise RuntimeError("mozyme_qmmm_init failed")
        f = self.lib.mozyme_qmmm_step
        f.argtypes = [C.c_int, C.POINTER(C.c_double), C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_double),
                      C.c_double, C.c_double, C.POINTER(C.c_double), C.POINTER(C.c_double),
                      C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
                      C.POINTER(C.c_double), C.POINTER(C.c_int)]
        f.restype = None

    def step(self, coords, mm_coords, mm_charges, ron=0.0, roff=0.0, qm_pot=None):
        x = np.ascontiguousarray(coords, float).ravel()
        vq = np.zeros(self.n) if qm_pot is None else np.ascontiguousarray(qm_pot, float).ravel()
        cq = np.zeros(self.n)
        mx = np.ascontiguousarray(mm_coords, float).ravel()
        mq = np.ascontiguousarray(mm_charges, float).ravel()
        nmm = len(mq)
        heat = C.c_double(0.0)
        g = np.zeros(3 * self.n)
        gm = np.zeros(max(3 * nmm, 1))
        pot = np.zeros(max(nmm, 1))
        status = C.c_int(1)
        dp = C.POINTER(C.c_double)
        self.lib.mozyme_qmmm_step(self.n, x.ctypes.data_as(dp), nmm, mx.ctypes.data_as(dp) if nmm else None,
                                  mq.ctypes.data_as(dp) if nmm else None, ron, roff, vq.ctypes.data_as(dp),
                                  C.byref(heat), g.ctypes.data_as(dp), gm.ctypes.data_as(dp),
                                  pot.ctypes.data_as(dp), cq.ctypes.data_as(dp), C.byref(status))
        if status.value == 2:
            raise RuntimeError("mozyme_qmmm_step: the SCF did not converge (tolerance too tight?)")
        if status.value != 0:
            raise RuntimeError("mozyme_qmmm_step failed: " + "; ".join(self.errors()))
        self.mm_pot = pot[:nmm]
        self.qm_charges = cq
        return heat.value, g.reshape(self.n, 3), gm[:3 * nmm].reshape(nmm, 3)

    def errors(self):
        """Finish the session and return MOPAC's error messages (after a failed step)."""
        props = MopacProperties()
        self.lib.mozyme_qmmm_finalize(C.byref(self.state), C.byref(props))
        msgs = []
        if props.nerror > 0 and props.error_msg:
            arr = C.cast(props.error_msg, C.POINTER(C.c_char_p))
            msgs = [arr[i].decode(errors="replace").strip() for i in range(props.nerror)]
        self.lib.destroy_mopac_properties(C.byref(props))
        self.closed = True
        return msgs

    def close(self):
        if getattr(self, "closed", False):
            return
        props = MopacProperties()
        self.lib.mozyme_qmmm_finalize(C.byref(self.state), C.byref(props))
        self.lib.destroy_mopac_properties(C.byref(props))
        self.lib.destroy_mozyme_state(C.byref(self.state))
