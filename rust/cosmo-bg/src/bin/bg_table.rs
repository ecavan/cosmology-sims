//! CSV of E, D_M, D_H, D_L, D_V on a redshift grid: bg_table [model=..] [h=..] [ombh2=..] ... [zs=0.1,0.5]

use cosmo_bg::cosmo_from_args;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let m = cosmo_from_args(&args);
    let zs: Vec<f64> = args
        .iter()
        .find_map(|a| a.strip_prefix("zs="))
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![0.01, 0.1, 0.3, 0.51, 0.706, 0.934, 1.321, 1.484, 2.33, 5.0, 10.0, 100.0, 1089.9]);
    println!("# zd_eh98={:.4} rd_integral={:.6}", m.z_drag_eh98(), m.r_d_integral());
    println!("z,E,D_M,D_H,D_L,D_V");
    for z in zs {
        println!("{z},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e}", m.e_of_z(z), m.d_m(z), m.d_h(z), m.d_l(z), m.d_v(z));
    }
}
