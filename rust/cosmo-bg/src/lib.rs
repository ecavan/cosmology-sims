//! Background cosmology for ΛCDM, w0wa (CPL), the running vacuum (RVM) and the flipped RVM.

pub mod constants;
pub mod distances;
pub mod effective;
pub mod model;
pub mod neutrino;
pub mod quad;

pub use model::{Cosmo, DarkEnergy};

/// Parse `key=value` arguments (h, ombh2, omch2, neff, tcmb, mnu, model, w0, wa, nu, z1, z2) into a Cosmo.
pub fn cosmo_from_args(args: &[String]) -> Cosmo {
    let mut c = Cosmo::planck18_massless();
    let (mut model, mut w0, mut wa, mut nu, mut z1, mut z2) = ("lcdm".to_string(), -1.0, 0.0, 0.0, 1.0, 2.0);
    for kv in args {
        let Some((k, v)) = kv.split_once('=') else { continue };
        let num = || v.parse::<f64>().unwrap_or_else(|_| panic!("bad number for {k}: {v}"));
        match k {
            "h" => c.h = num(),
            "ombh2" => c.omega_b = num(),
            "omch2" => c.omega_c = num(),
            "neff" => c.n_eff = num(),
            "tcmb" => c.t_cmb = num(),
            "mnu" => c.m_nu = num(),
            "w0" => w0 = num(),
            "wa" => wa = num(),
            "nu" => nu = num(),
            "z1" => z1 = num(),
            "z2" => z2 = num(),
            "model" => model = v.to_string(),
            _ => {}
        }
    }
    c.de = DarkEnergy::from_name(&model, w0, wa, nu, z1, z2).unwrap_or_else(|e| panic!("{e}"));
    c
}
