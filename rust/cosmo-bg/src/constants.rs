//! Physical constants (CODATA 2018, SI) and derived cosmological densities.

/// Speed of light [km/s].
pub const C_KM_S: f64 = 299_792.458;
/// Speed of light [m/s].
pub const C_M_S: f64 = 299_792_458.0;
/// Newton's constant [m^3 kg^-1 s^-2].
pub const G_SI: f64 = 6.674_30e-11;
/// Megaparsec [m].
pub const MPC_M: f64 = 3.085_677_581_491_367_3e22;
/// Stefan–Boltzmann constant [W m^-2 K^-4].
pub const SIGMA_SB: f64 = 5.670_374_419e-8;

/// (7/8)(4/11)^(4/3): energy density of one massless neutrino species relative to photons.
pub const NU_PER_SPECIES: f64 = 0.227_107_317_660_239_4;

/// Critical energy density ρc² for H0 = 100 km/s/Mpc [J/m³].
pub fn rho_crit_100_energy() -> f64 {
    let h100_si = 100.0e3 / MPC_M; // s^-1
    3.0 * h100_si * h100_si / (8.0 * std::f64::consts::PI * G_SI) * C_M_S * C_M_S
}

/// Photon density ω_γ = Ω_γ h² for CMB temperature t_cmb [K].
pub fn omega_gamma(t_cmb: f64) -> f64 {
    let a_rad = 4.0 * SIGMA_SB / C_M_S; // radiation constant [J m^-3 K^-4]
    a_rad * t_cmb.powi(4) / rho_crit_100_energy()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn photon_density_matches_standard_value() {
        let og = omega_gamma(2.7255);
        assert!((og / 2.4729e-5 - 1.0).abs() < 2e-4, "omega_gamma = {og}");
    }
    #[test]
    fn neutrino_factor() {
        let exact = 7.0 / 8.0 * (4.0f64 / 11.0).powf(4.0 / 3.0);
        assert!((exact - NU_PER_SPECIES).abs() < 1e-15);
    }
}
