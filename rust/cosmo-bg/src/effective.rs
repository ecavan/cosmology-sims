//! What dark energy an observer infers when assuming matter ∝ a⁻³ anchored at the CMB.
//!
//! ρ̃_DE(a) = E²(a) − Ω̃_m a⁻³ − Ω_r a⁻⁴ − Ω_ν,m(a) with Ω̃_m = Ω_m(a*) a*³, and w_eff = −1 − (1/3) dln ρ̃_DE/dln a.

use crate::model::Cosmo;

/// Recombination redshift used as the CMB anchor.
pub const Z_STAR: f64 = 1090.0;

/// Observer's Ω̃_m: true matter density at the anchor, scaled to today with a³.
pub fn anchored_omega_m(m: &Cosmo, z_anchor: f64) -> f64 {
    let a = 1.0 / (1.0 + z_anchor);
    m.densities(a).matter() * a.powi(3)
}

/// Effective dark-energy density seen by the observer.
pub fn rho_de_eff(m: &Cosmo, a: f64, om_tilde: f64) -> f64 {
    m.e2_of_a(a) - om_tilde * a.powi(-3) - m.omegas().r * a.powi(-4) - m.nu_massive(a)
}

/// Effective equation of state (central difference in ln a).
pub fn w_eff(m: &Cosmo, a: f64, om_tilde: f64) -> f64 {
    let (x, h) = (a.ln(), 1e-4);
    let lp = rho_de_eff(m, (x + h).exp(), om_tilde).ln();
    let lm = rho_de_eff(m, (x - h).exp(), om_tilde).ln();
    -1.0 - (lp - lm) / (2.0 * h) / 3.0
}

#[derive(Clone, Copy, Debug)]
pub struct CplFit {
    pub w0: f64,
    pub wa: f64,
    /// RMS residual of ln ρ̃_DE about the fit.
    pub rms: f64,
}

/// Linear least-squares CPL fit to ln ρ̃_DE on n points in (0, z_max]; None if ρ̃_DE ≤ 0 anywhere.
pub fn fit_cpl(m: &Cosmo, om_tilde: f64, z_max: f64, n: usize) -> Option<CplFit> {
    let rho0 = rho_de_eff(m, 1.0, om_tilde);
    if rho0 <= 0.0 {
        return None;
    }
    let mut rows = Vec::with_capacity(n);
    for i in 1..=n {
        let a = 1.0 / (1.0 + z_max * i as f64 / n as f64);
        let rho = rho_de_eff(m, a, om_tilde);
        if rho <= 0.0 {
            return None;
        }
        let la = a.ln();
        // ln(ρ̃/ρ̃0) = p L + q M with p = −3(1+w0), q = −3wa, L = ln a, M = ln a + 1 − a
        rows.push((la, la + 1.0 - a, (rho / rho0).ln()));
    }
    let (mut sll, mut slm, mut smm, mut sly, mut smy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(l, mm, y) in &rows {
        sll += l * l;
        slm += l * mm;
        smm += mm * mm;
        sly += l * y;
        smy += mm * y;
    }
    let det = sll * smm - slm * slm;
    let p = (sly * smm - smy * slm) / det;
    let q = (smy * sll - sly * slm) / det;
    let rms = (rows.iter().map(|&(l, mm, y)| (y - p * l - q * mm).powi(2)).sum::<f64>() / n as f64).sqrt();
    Some(CplFit { w0: -p / 3.0 - 1.0, wa: -q / 3.0, rms })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DarkEnergy;

    #[test]
    fn lcdm_maps_to_lambda() {
        let m = Cosmo::planck18_massless();
        let om = anchored_omega_m(&m, Z_STAR);
        assert!((om - (m.omegas().b + m.omegas().c)).abs() < 1e-9);
        let f = fit_cpl(&m, om, 2.5, 200).unwrap();
        assert!((f.w0 + 1.0).abs() < 1e-9 && f.wa.abs() < 1e-9, "{f:?}");
        assert!((w_eff(&m, 0.5, om) + 1.0).abs() < 1e-7);
    }

    /// A CPL truth with standard matter is recovered exactly.
    #[test]
    fn cpl_truth_is_recovered() {
        let (w0, wa) = (-0.75, -0.85);
        let m = Cosmo::planck18_massless().with_de(DarkEnergy::Cpl { w0, wa });
        let om = anchored_omega_m(&m, Z_STAR);
        let f = fit_cpl(&m, om, 2.5, 200).unwrap();
        assert!((f.w0 - w0).abs() < 1e-9 && (f.wa - wa).abs() < 1e-9, "{f:?}");
        assert!((w_eff(&m, 0.6, om) - (w0 + wa * 0.4)).abs() < 1e-7);
    }
}
