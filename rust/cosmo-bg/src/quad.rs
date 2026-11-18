//! Adaptive Gauss–Kronrod (G7/K15) quadrature; nodes are interior, so endpoint singularities are never evaluated.

const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
const WGK: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_2,
    0.140_653_259_715_525_9,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_8,
];
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

fn gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fc = f(c);
    let mut rk = fc * WGK[7];
    let mut rg = fc * WG[3];
    for j in 0..7 {
        let x = h * XGK[j];
        let s = f(c - x) + f(c + x);
        rk += WGK[j] * s;
        if j % 2 == 1 {
            rg += WG[j / 2] * s;
        }
    }
    (rk * h, ((rk - rg) * h).abs())
}

/// Integrate f over [a, b] to relative tolerance rel_tol.
pub fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, rel_tol: f64) -> f64 {
    fn rec<F: Fn(f64) -> f64>(
        f: &F,
        a: f64,
        b: f64,
        whole: f64,
        err: f64,
        tol: f64,
        depth: u32,
    ) -> f64 {
        if err <= tol || depth > 60 {
            return whole;
        }
        let m = 0.5 * (a + b);
        let (l, el) = gk15(f, a, m);
        let (r, er) = gk15(f, m, b);
        rec(f, a, m, l, el, 0.5 * tol, depth + 1) + rec(f, m, b, r, er, 0.5 * tol, depth + 1)
    }
    if a == b {
        return 0.0;
    }
    let (whole, err) = gk15(&f, a, b);
    let tol = (rel_tol * whole.abs()).max(1e-300);
    rec(&f, a, b, whole, err, tol, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn polynomial_and_transcendental() {
        assert!((integrate(|x| x * x, 0.0, 3.0, 1e-12) - 9.0).abs() < 1e-12);
        let v = integrate(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-12);
        assert!((v - 2.0).abs() < 1e-12);
        // integrable endpoint singularity
        let s = integrate(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10);
        assert!((s - 2.0).abs() < 1e-8, "{s}");
    }
}
