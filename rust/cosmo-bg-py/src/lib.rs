//! Python bindings: `cosmo_bg.Cosmology(model="rvm", nu=1e-3).d_m([0.5, 1.0])`.

use ::cosmo_bg as core;
use core::effective::{anchored_omega_m, fit_cpl, w_eff, Z_STAR};
use core::{Cosmo, DarkEnergy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name = "Cosmology")]
#[derive(Clone)]
struct PyCosmo {
    inner: Cosmo,
}

#[pymethods]
impl PyCosmo {
    #[new]
    #[pyo3(signature = (h=0.6736, ombh2=0.02237, omch2=0.1200, neff=3.044, tcmb=2.7255, model="lcdm", w0=-1.0, wa=0.0, nu=0.0, m_nu=0.0, z1=1.0, z2=2.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(h: f64, ombh2: f64, omch2: f64, neff: f64, tcmb: f64, model: &str, w0: f64, wa: f64, nu: f64, m_nu: f64,
           z1: f64, z2: f64) -> PyResult<Self> {
        let de = DarkEnergy::from_name(model, w0, wa, nu, z1, z2).map_err(PyValueError::new_err)?;
        Ok(PyCosmo { inner: Cosmo { h, omega_b: ombh2, omega_c: omch2, t_cmb: tcmb, n_eff: neff, m_nu, de } })
    }

    fn e_of_z(&self, zs: Vec<f64>) -> Vec<f64> {
        zs.iter().map(|&z| self.inner.e_of_z(z)).collect()
    }
    fn d_m(&self, zs: Vec<f64>) -> Vec<f64> {
        self.inner.d_c_many(&zs)
    }
    fn d_h(&self, zs: Vec<f64>) -> Vec<f64> {
        zs.iter().map(|&z| self.inner.d_h(z)).collect()
    }
    fn d_l(&self, zs: Vec<f64>) -> Vec<f64> {
        zs.iter().zip(self.inner.d_c_many(&zs)).map(|(&z, d)| (1.0 + z) * d).collect()
    }
    fn d_v(&self, zs: Vec<f64>) -> Vec<f64> {
        zs.iter().map(|&z| self.inner.d_v(z)).collect()
    }
    fn z_drag_eh98(&self) -> f64 {
        self.inner.z_drag_eh98()
    }
    /// Sound horizon r_s(z) [Mpc] without recombination physics (use ratios).
    fn sound_horizon(&self, z: f64) -> f64 {
        self.inner.sound_horizon(z)
    }
    fn r_d_integral(&self) -> f64 {
        self.inner.r_d_integral()
    }
    fn omegas(&self) -> HashMap<&'static str, f64> {
        let o = self.inner.omegas();
        HashMap::from([("b", o.b), ("c", o.c), ("gamma", o.gamma), ("nu", o.nu), ("r", o.r), ("nu_m", o.nu_m), ("de", o.de)])
    }
    /// Observer's effective w(z) with matter anchored at z_anchor.
    #[pyo3(signature = (zs, z_anchor=Z_STAR))]
    fn w_eff(&self, zs: Vec<f64>, z_anchor: f64) -> Vec<f64> {
        let om = anchored_omega_m(&self.inner, z_anchor);
        zs.iter().map(|&z| w_eff(&self.inner, 1.0 / (1.0 + z), om)).collect()
    }
    /// CPL projection (w0, wa, rms) of the observer's dark energy over (0, z_max].
    #[pyo3(signature = (z_max=2.5, n=400, z_anchor=Z_STAR))]
    fn fit_cpl(&self, z_max: f64, n: usize, z_anchor: f64) -> Option<(f64, f64, f64)> {
        let om = anchored_omega_m(&self.inner, z_anchor);
        fit_cpl(&self.inner, om, z_max, n).map(|f| (f.w0, f.wa, f.rms))
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pymodule]
fn cosmo_bg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCosmo>()?;
    Ok(())
}
