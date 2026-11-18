//! One massive neutrino (CLASS convention, T_ncdm = 0.71611 T_cmb, ΔN_eff = 1.01317): ρ_ν = ρ_ν,rel · F(y).
//!
//! F(y) = ∫x²√(x²+y²)/(eˣ+1)dx / ∫x³/(eˣ+1)dx with y = m a/(k_B T_ncdm,0), tabulated in ln y (error < 1e-7).

use crate::quad::integrate;
use std::sync::OnceLock;

pub const T_NCDM_OVER_TCMB: f64 = 0.71611;
pub const DELTA_NEFF_ONE_MASSIVE: f64 = 1.01317;
const K_B_EV: f64 = 8.617_333_262e-5;

const Y_MIN: f64 = 1e-4;
const Y_MAX: f64 = 1e6;
const NTAB: usize = 3000;

fn f_direct(y: f64) -> f64 {
    let num = integrate(|x: f64| x * x * (x * x + y * y).sqrt() / (x.exp() + 1.0), 0.0, 80.0, 1e-12);
    num / (7.0 * std::f64::consts::PI.powi(4) / 120.0)
}

fn table() -> &'static Vec<f64> {
    static T: OnceLock<Vec<f64>> = OnceLock::new();
    T.get_or_init(|| {
        let (l0, l1) = (Y_MIN.ln(), Y_MAX.ln());
        (0..NTAB).map(|i| f_direct((l0 + (l1 - l0) * i as f64 / (NTAB - 1) as f64).exp()).ln()).collect()
    })
}

/// Energy-density ratio to the relativistic limit, F(y).
pub fn f_ratio(y: f64) -> f64 {
    if y <= Y_MIN {
        return 1.0; // F = 1 + O(y²)
    }
    if y >= Y_MAX {
        return 0.317_321_9 * y; // (3ζ(3)/2)/(7π⁴/120) · y
    }
    let (l0, l1) = (Y_MIN.ln(), Y_MAX.ln());
    let s = (y.ln() - l0) / (l1 - l0) * (NTAB - 1) as f64;
    let i = (s.floor() as usize).min(NTAB - 2);
    let w = s - i as f64;
    let t = table();
    ((1.0 - w) * t[i] + w * t[i + 1]).exp()
}

/// y = m a / (k_B T_ncdm,0) for mass m [eV].
pub fn y_of(m_ev: f64, a: f64, t_cmb: f64) -> f64 {
    m_ev * a / (K_B_EV * T_NCDM_OVER_TCMB * t_cmb)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn limits_and_interpolation() {
        assert!((f_direct(1e-6) - 1.0).abs() < 1e-9);
        let y = 3000.0;
        assert!((f_direct(y) / (0.3173219 * y) - 1.0).abs() < 1e-5);
        for &y in &[0.003, 0.7, 2.5, 37.0, 812.0] {
            assert!((f_ratio(y) / f_direct(y) - 1.0).abs() < 1e-6, "y={y}");
        }
    }
    #[test]
    fn omega_nu_today_matches_93_14_ev_rule() {
        // ω_ν = m/93.14 eV for one non-relativistic species
        let t = 2.7255;
        let og = crate::constants::omega_gamma(t);
        let rel = og * 7.0 / 8.0 * T_NCDM_OVER_TCMB.powi(4);
        let onu = rel * f_ratio(y_of(0.06, 1.0, t));
        assert!((onu / (0.06 / 93.14) - 1.0).abs() < 0.005, "ω_ν = {onu}");
    }
}
