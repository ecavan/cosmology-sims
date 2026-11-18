//! Flat FLRW background; densities in units of today's critical density, so E²(a) is their sum.
//!
//! Running vacuum: Ω_vac = Ω_vac0 + ν(E² − 1), exchanging energy with CDM only, which gives
//! ρ_c(a) = A a^{−3(1−ν)} − ρ_b0 a^{−3} − [4ν/(1+3ν)] ρ_r0 a^{−4}, A = ρ_c0 + ρ_b0 + 4νρ_r0/(1+3ν).

use crate::constants::{omega_gamma, NU_PER_SPECIES};
use crate::neutrino::{f_ratio, y_of, DELTA_NEFF_ONE_MASSIVE, T_NCDM_OVER_TCMB};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DarkEnergy {
    /// Cosmological constant.
    Lambda,
    /// CPL: w(a) = w0 + wa (1 − a).
    Cpl { w0: f64, wa: f64 },
    /// Running vacuum ρ_vac = ρ_vac0 + ν(ρ_crit(H) − ρ_crit0), exchanging energy with CDM.
    RunningVacuum { nu: f64 },
    /// Flipped RVM (arXiv:2512.20616 §II.E): matter ∝ a^{−3(1−ν)} for a > a1, a^{−3(1+ν)} for a2 < a < a1, a^{−3} below.
    Flipped { nu: f64, a1: f64, a2: f64 },
}

impl DarkEnergy {
    /// Build from a model name: lcdm | cpl | rvm | flipped (z1, z2 are the flipped switch redshifts).
    pub fn from_name(model: &str, w0: f64, wa: f64, nu: f64, z1: f64, z2: f64) -> Result<Self, String> {
        Ok(match model {
            "lcdm" => DarkEnergy::Lambda,
            "cpl" => DarkEnergy::Cpl { w0, wa },
            "rvm" => DarkEnergy::RunningVacuum { nu },
            "flipped" => DarkEnergy::Flipped { nu, a1: 1.0 / (1.0 + z1), a2: 1.0 / (1.0 + z2) },
            other => return Err(format!("unknown model {other}: use lcdm | cpl | rvm | flipped")),
        })
    }
}

/// Flat cosmology given by physical densities ω = Ω h² today.
#[derive(Clone, Copy, Debug)]
pub struct Cosmo {
    pub h: f64,
    pub omega_b: f64,
    pub omega_c: f64,
    pub t_cmb: f64,
    /// Total effective number of neutrino species (CLASS convention).
    pub n_eff: f64,
    /// Mass of one massive neutrino [eV]; 0 means all massless, else N_ur = n_eff − 1.01317.
    pub m_nu: f64,
    pub de: DarkEnergy,
}

/// Today's density parameters.
#[derive(Clone, Copy, Debug)]
pub struct Omegas {
    pub b: f64,
    pub c: f64,
    pub gamma: f64,
    pub nu: f64,
    pub r: f64,
    pub nu_m: f64,
    pub de: f64,
}

/// Densities at scale factor a (units of today's critical density).
#[derive(Clone, Copy, Debug)]
pub struct Densities {
    pub b: f64,
    pub c: f64,
    pub r: f64,
    pub nu_m: f64,
    pub de: f64,
}

impl Densities {
    pub fn total(&self) -> f64 {
        self.b + self.c + self.r + self.nu_m + self.de
    }
    pub fn matter(&self) -> f64 {
        self.b + self.c
    }
}

impl Cosmo {
    /// Planck-2018-like ΛCDM with massless neutrinos (N_eff = 3.044).
    pub fn planck18_massless() -> Self {
        Cosmo { h: 0.6736, omega_b: 0.02237, omega_c: 0.1200, t_cmb: 2.7255, n_eff: 3.044, m_nu: 0.0, de: DarkEnergy::Lambda }
    }

    pub fn with_de(mut self, de: DarkEnergy) -> Self {
        self.de = de;
        self
    }

    /// Massive-neutrino density at scale factor a.
    pub fn nu_massive(&self, a: f64) -> f64 {
        if self.m_nu <= 0.0 {
            return 0.0;
        }
        let rel = omega_gamma(self.t_cmb) / (self.h * self.h) * 7.0 / 8.0 * T_NCDM_OVER_TCMB.powi(4);
        rel * a.powi(-4) * f_ratio(y_of(self.m_nu, a, self.t_cmb))
    }

    pub fn omegas(&self) -> Omegas {
        let h2 = self.h * self.h;
        let gamma = omega_gamma(self.t_cmb) / h2;
        let n_ur = if self.m_nu > 0.0 { self.n_eff - DELTA_NEFF_ONE_MASSIVE } else { self.n_eff };
        let nu = gamma * NU_PER_SPECIES * n_ur;
        let (b, c, r, nu_m) = (self.omega_b / h2, self.omega_c / h2, gamma + nu, self.nu_massive(1.0));
        Omegas { b, c, gamma, nu, r, nu_m, de: 1.0 - b - c - r - nu_m }
    }

    /// Hubble constant [km/s/Mpc].
    pub fn h0(&self) -> f64 {
        100.0 * self.h
    }

    pub fn densities(&self, a: f64) -> Densities {
        let o = self.omegas();
        let (b, r, nu_m) = (o.b * a.powi(-3), o.r * a.powi(-4), self.nu_massive(a));
        let (c, de) = match self.de {
            DarkEnergy::Lambda => (o.c * a.powi(-3), o.de),
            DarkEnergy::Cpl { w0, wa } => {
                (o.c * a.powi(-3), o.de * a.powf(-3.0 * (1.0 + w0 + wa)) * (-3.0 * wa * (1.0 - a)).exp())
            }
            DarkEnergy::RunningVacuum { nu } => {
                let k = 4.0 * nu / (1.0 + 3.0 * nu);
                let amp = o.c + o.b + k * o.r;
                let c = amp * a.powf(-3.0 * (1.0 - nu)) - o.b * a.powi(-3) - k * o.r * a.powi(-4);
                // Friedmann with de = Ω_vac0 + ν(E² − 1); the massive-ν source term (O(νΩ_ν)) is neglected.
                let e2 = (c + b + r + nu_m + o.de - nu) / (1.0 - nu);
                (c, o.de + nu * (e2 - 1.0))
            }
            DarkEnergy::Flipped { nu, a1, a2 } => {
                let (rm, vac) = flipped_matter_vacuum(o.b + o.c, o.de, nu, a1, a2, a);
                (rm - b, vac)
            }
        };
        Densities { b, c, r, nu_m, de }
    }

    /// E²(a) = H²/H0².
    pub fn e2_of_a(&self, a: f64) -> f64 {
        self.densities(a).total()
    }

    pub fn e_of_z(&self, z: f64) -> f64 {
        self.e2_of_a(1.0 / (1.0 + z)).sqrt()
    }
}

/// Flipped RVM matter and vacuum densities; in each segment ρ_vac − ρ_vac(a_e) = ((3 − n)/n)(ρ_m − ρ_m(a_e)).
pub fn flipped_matter_vacuum(om0: f64, vac0: f64, nu: f64, a1: f64, a2: f64, a: f64) -> (f64, f64) {
    let (n1, n2) = (3.0 * (1.0 - nu), 3.0 * (1.0 + nu));
    let m1 = om0 * a1.powf(-n1);
    let v1 = vac0 + (3.0 - n1) / n1 * (m1 - om0);
    let m2 = m1 * (a2 / a1).powf(-n2);
    let v2 = v1 + (3.0 - n2) / n2 * (m2 - m1);
    if a >= a1 {
        let m = om0 * a.powf(-n1);
        (m, vac0 + (3.0 - n1) / n1 * (m - om0))
    } else if a >= a2 {
        let m = m1 * (a / a1).powf(-n2);
        (m, v1 + (3.0 - n2) / n2 * (m - m1))
    } else {
        (m2 * (a / a2).powi(-3), v2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Cosmo {
        Cosmo::planck18_massless()
    }

    /// Fourth-order Runge–Kutta in x = ln a from x = 0 to ln(1e-4).
    fn rk4(rhs: impl Fn(f64, f64) -> f64, y0: f64) -> (f64, f64) {
        let (x_end, n) = ((1e-4f64).ln(), 200_000);
        let dx = x_end / n as f64;
        let (mut x, mut y) = (0.0, y0);
        for _ in 0..n {
            let k1 = rhs(x, y);
            let k2 = rhs(x + 0.5 * dx, y + 0.5 * dx * k1);
            let k3 = rhs(x + 0.5 * dx, y + 0.5 * dx * k2);
            let k4 = rhs(x + dx, y + dx * k3);
            y += dx / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            x += dx;
        }
        (x, y)
    }

    #[test]
    fn flat_closure_today() {
        for de in [
            DarkEnergy::Lambda,
            DarkEnergy::Cpl { w0: -0.8, wa: -0.7 },
            DarkEnergy::RunningVacuum { nu: 3e-3 },
            DarkEnergy::Flipped { nu: 0.03, a1: 0.45, a2: 0.3 },
        ] {
            let e2 = base().with_de(de).e2_of_a(1.0);
            assert!((e2 - 1.0).abs() < 1e-14, "{de:?}: E2(1) = {e2}");
        }
    }

    #[test]
    fn rvm_nu_zero_is_lcdm() {
        let (l, r) = (base(), base().with_de(DarkEnergy::RunningVacuum { nu: 0.0 }));
        for &a in &[1e-8, 1e-4, 1e-3, 0.1, 0.5, 1.0] {
            let (x, y) = (l.e2_of_a(a), r.e2_of_a(a));
            assert!((x / y - 1.0).abs() < 1e-13, "a={a}: {x} vs {y}");
        }
    }

    #[test]
    fn cpl_lambda_limit() {
        let (l, c) = (base(), base().with_de(DarkEnergy::Cpl { w0: -1.0, wa: 0.0 }));
        for &a in &[1e-4, 0.3, 1.0] {
            assert!((l.e2_of_a(a) / c.e2_of_a(a) - 1.0).abs() < 1e-14);
        }
    }

    /// Closed-form ρ_c(a) agrees with RK4 of dρ_c/dln a = −3(1−ν)ρ_c + ν(3ρ_b + 4ρ_r).
    #[test]
    fn rvm_closed_form_matches_ode() {
        let nu = 5e-3;
        let m = base().with_de(DarkEnergy::RunningVacuum { nu });
        let o = m.omegas();
        let (x, rc) = rk4(|x, rc| {
            let a = f64::exp(x);
            -3.0 * (1.0 - nu) * rc + nu * (3.0 * o.b * a.powi(-3) + 4.0 * o.r * a.powi(-4))
        }, o.c);
        let exact = m.densities(x.exp()).c;
        assert!((rc / exact - 1.0).abs() < 1e-9, "ode {rc} vs closed {exact}");
    }

    /// Energy conservation dρ_vac/dln a = −(dρ_c/dln a + 3ρ_c), by finite differences.
    #[test]
    fn rvm_exchange_conserves_energy() {
        let m = base().with_de(DarkEnergy::RunningVacuum { nu: 4e-3 });
        for &a in &[1e-3, 0.05, 0.4, 0.9] {
            let (x, hs) = (f64::ln(a), 1e-5);
            let (p, q) = (m.densities((x + hs).exp()), m.densities((x - hs).exp()));
            let (dvac, dc) = ((p.de - q.de) / (2.0 * hs), (p.c - q.c) / (2.0 * hs));
            let c = m.densities(a).c;
            let resid = dvac + dc + 3.0 * c;
            assert!(resid.abs() < 1e-7 * (dc.abs() + 3.0 * c), "a={a}: residual {resid}");
        }
    }

    /// The vacuum obeys Ω_vac(a) = Ω_vac0 + ν(E² − 1).
    #[test]
    fn rvm_vacuum_tracks_h2() {
        let nu = -2e-3;
        let m = base().with_de(DarkEnergy::RunningVacuum { nu });
        let o = m.omegas();
        for &a in &[1e-3, 0.2, 0.7] {
            let d = m.densities(a);
            assert!((d.de - (o.de + nu * (d.total() - 1.0))).abs() < 1e-12);
        }
    }

    #[test]
    fn flipped_lcdm_limit() {
        let (l, f) = (base(), base().with_de(DarkEnergy::Flipped { nu: 0.0, a1: 0.5, a2: 0.3 }));
        for &a in &[1e-6, 1e-3, 0.2, 0.4, 0.7, 1.0] {
            assert!((f.e2_of_a(a) / l.e2_of_a(a) - 1.0).abs() < 1e-13);
        }
    }

    /// Flipped RVM: continuity at a1, a2 and energy conservation inside each segment.
    #[test]
    fn flipped_continuity_and_conservation() {
        let (a1, a2) = (0.45, 0.3);
        let m = base().with_de(DarkEnergy::Flipped { nu: 0.03, a1, a2 });
        for &ae in &[a1, a2] {
            let (p, q) = (m.densities(ae * (1.0 + 1e-12)), m.densities(ae * (1.0 - 1e-12)));
            assert!((p.c / q.c - 1.0).abs() < 1e-9 && (p.de / q.de - 1.0).abs() < 1e-9);
        }
        for &a in &[0.1, 0.2, 0.35, 0.4, 0.6, 0.95] {
            let (x, hs) = (f64::ln(a), 1e-6);
            let (p, q) = (m.densities((x + hs).exp()), m.densities((x - hs).exp()));
            let (dvac, dc) = ((p.de - q.de) / (2.0 * hs), (p.c - q.c) / (2.0 * hs));
            let c = m.densities(a).c;
            assert!((dvac + dc + 3.0 * c).abs() < 1e-6 * (dc.abs() + 3.0 * c), "a={a}");
        }
    }
}
