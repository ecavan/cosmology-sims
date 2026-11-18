//! Flat-universe distances in Mpc (Hogg 1999) and the sound horizon.

use crate::constants::{omega_gamma, C_KM_S};
use crate::model::Cosmo;
use crate::quad::integrate;

const REL_TOL: f64 = 1e-11;

impl Cosmo {
    /// c/H0 [Mpc].
    pub fn hubble_distance(&self) -> f64 {
        C_KM_S / self.h0()
    }

    /// D_H(z) = c/H(z).
    pub fn d_h(&self, z: f64) -> f64 {
        self.hubble_distance() / self.e_of_z(z)
    }

    /// D_C(z), integrated in x = ln(1+z), where the integrand (1+z)/E is smooth up to recombination.
    pub fn d_c(&self, z: f64) -> f64 {
        self.hubble_distance() * integrate(|x: f64| x.exp() / self.e_of_z(x.exp_m1()), 0.0, z.ln_1p(), REL_TOL)
    }

    /// D_C at many redshifts: cumulative integrals for ≤ 32 points, else a cubic-Hermite table in ln(1+z).
    pub fn d_c_many(&self, zs: &[f64]) -> Vec<f64> {
        let f = |y: f64| y.exp() / self.e_of_z(y.exp_m1());
        let dh = self.hubble_distance();
        if zs.len() <= 32 {
            let mut idx: Vec<usize> = (0..zs.len()).collect();
            idx.sort_by(|&i, &j| zs[i].partial_cmp(&zs[j]).unwrap());
            let mut out = vec![0.0; zs.len()];
            let (mut x_prev, mut acc) = (0.0f64, 0.0f64);
            for &i in &idx {
                let x = zs[i].ln_1p();
                if x > x_prev {
                    acc += integrate(f, x_prev, x, REL_TOL);
                    x_prev = x;
                }
                out[i] = dh * acc;
            }
            return out;
        }
        // Grid spacing ≤ 0.002 in x with exact derivatives: error ≲ 1e-9 in D_C (tested below).
        let x_max = zs.iter().cloned().fold(0.0f64, f64::max).ln_1p();
        let n = ((x_max / 0.002).ceil() as usize).max(4);
        let h = x_max / n as f64;
        let (mut val, mut der, mut acc) = (Vec::with_capacity(n + 1), Vec::with_capacity(n + 1), 0.0);
        for k in 0..=n {
            let x = k as f64 * h;
            if k > 0 {
                acc += integrate(f, x - h, x, REL_TOL);
            }
            val.push(acc);
            der.push(f(x));
        }
        zs.iter()
            .map(|&z| {
                let x = z.ln_1p();
                let k = ((x / h).floor() as usize).min(n - 1);
                let t = (x - k as f64 * h) / h;
                let (t2, t3) = (t * t, t * t * t);
                let (h00, h10, h01, h11) = (2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, -2.0 * t3 + 3.0 * t2, t3 - t2);
                dh * (h00 * val[k] + h10 * h * der[k] + h01 * val[k + 1] + h11 * h * der[k + 1])
            })
            .collect()
    }

    /// D_M(z) = D_C(z) (flat).
    pub fn d_m(&self, z: f64) -> f64 {
        self.d_c(z)
    }

    pub fn d_l(&self, z: f64) -> f64 {
        (1.0 + z) * self.d_m(z)
    }

    /// D_V = [z D_M² D_H]^(1/3).
    pub fn d_v(&self, z: f64) -> f64 {
        (z * self.d_m(z).powi(2) * self.d_h(z)).cbrt()
    }

    /// Drag redshift, Eisenstein & Hu (1998) eq. 4.
    pub fn z_drag_eh98(&self) -> f64 {
        let (om, ob) = (self.omega_b + self.omega_c, self.omega_b);
        let b1 = 0.313 * om.powf(-0.419) * (1.0 + 0.607 * om.powf(0.674));
        let b2 = 0.238 * om.powf(0.223);
        1291.0 * om.powf(0.251) / (1.0 + 0.659 * om.powf(0.828)) * (1.0 + b1 * ob.powf(b2))
    }

    /// r_s(z) = ∫_z^∞ c_s dz/H with c_s = c/√(3(1+R)), R = 3ρ_b/4ρ_γ, integrated in a.
    pub fn sound_horizon(&self, z: f64) -> f64 {
        let r_coeff = 3.0 * self.omega_b / (4.0 * omega_gamma(self.t_cmb));
        let integrand = |a: f64| {
            let cs = 1.0 / (3.0 * (1.0 + r_coeff * a)).sqrt();
            cs / (a * a * self.e2_of_a(a).sqrt())
        };
        self.hubble_distance() * integrate(integrand, 0.0, 1.0 / (1.0 + z), REL_TOL)
    }

    /// r_s at the EH98 drag redshift; ~2% above CLASS/CAMB, so the analysis uses only ratios of it.
    pub fn r_d_integral(&self) -> f64 {
        self.sound_horizon(self.z_drag_eh98())
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{Cosmo, DarkEnergy};
    use crate::quad::integrate;

    #[test]
    fn low_z_hubble_law() {
        let m = Cosmo::planck18_massless();
        let z = 1e-4;
        assert!((m.d_c(z) / (m.hubble_distance() * z) - 1.0).abs() < 1e-4);
    }

    /// d_c and both d_c_many paths agree with direct integration in z.
    #[test]
    fn d_c_variants_agree() {
        for de in [DarkEnergy::Lambda, DarkEnergy::Cpl { w0: -0.8, wa: -0.7 }, DarkEnergy::Flipped { nu: 0.03, a1: 0.45, a2: 0.3 }] {
            let m = Cosmo { m_nu: 0.06, ..Cosmo::planck18_massless() }.with_de(de);
            let direct = |z: f64| m.hubble_distance() * integrate(|zp| 1.0 / m.e_of_z(zp), 0.0, z, 1e-12);
            let zs = [1.5, 0.01, 0.3, 2.33, 0.3, 1089.9, 0.7, 1e-3];
            let many = m.d_c_many(&zs);
            for (k, &z) in zs.iter().enumerate() {
                assert!((m.d_c(z) / direct(z) - 1.0).abs() < 1e-10, "{de:?} z={z}");
                assert!((many[k] / direct(z) - 1.0).abs() < 1e-10, "{de:?} z={z} (many)");
            }
            // SN-like list (Hermite path) with duplicates and endpoints
            let zl: Vec<f64> = (0..400).map(|k| 1e-3 + 2.3 * ((k * 37 % 400) as f64 / 399.0).powi(2)).collect();
            let worst_mu = m.d_c_many(&zl).iter().zip(&zl)
                .map(|(d, &z)| 5.0 * (1.0 + (d / direct(z) - 1.0).abs()).log10())
                .fold(0.0f64, f64::max);
            assert!(worst_mu < 1e-9, "{de:?}: Hermite error in μ {worst_mu:.2e} mag");
        }
    }

    /// EH98 puts z_d ≈ 1020 for Planck parameters (CLASS ≈ 1060), so the raw r_d overshoots.
    #[test]
    fn sound_horizon_in_expected_range() {
        let m = Cosmo::planck18_massless();
        assert!((1015.0..1025.0).contains(&m.z_drag_eh98()));
        assert!((150.0..151.5).contains(&m.r_d_integral()));
    }
}
