//! Native hot-loop spike (PH3-WI02): closed static net identical in workload to
//! the Python B003 benchmark (StaticNetSpec 256/300/degree-8/seed-9), with the
//! D2.3 keyed stream reimplemented over blake2b-8 so drive and fan-out match
//! the oracle bit-for-bit. Prints total spikes, full-precision final-voltage
//! checksum, and wall time. Runs twice and asserts internal determinism.

use blake2::{Blake2b, Digest};
use digest::consts::U8;
use std::collections::HashMap;
use std::time::Instant;

// StaticNetSpec(256, 300, out_degree=8, weight=0.4, delay=2, seed=9,
//               density=0.4, amplitude=3.0) with default LIFParams.
const N: usize = 256;
const TICKS: usize = 300;
const DEGREE: usize = 8;
const WEIGHT: f64 = 0.5 - 0.1;
const DELAY: usize = 2;
const SEED: u64 = 9;
const DENSITY: f64 = 0.4;
const AMPLITUDE: f64 = 3.0;
const DRIVE_BLOCK: usize = 100;
const DRIVE_POP: u64 = 0xD21E;
const TAU: f64 = 20.0;
const DT: f64 = 0.1;
const THRESHOLD: f64 = 1.0;
const V_RESET: f64 = 0.0;
const REF_TICKS: i64 = 20;

fn keyed_int(seed: u64, source: u64, pop: u64, version: u64, salt: u32, domain: u32) -> u64 {
    let mut bytes = [0u8; 40];
    bytes[0..8].copy_from_slice(&seed.to_le_bytes());
    bytes[8..16].copy_from_slice(&source.to_le_bytes());
    bytes[16..24].copy_from_slice(&pop.to_le_bytes());
    bytes[24..32].copy_from_slice(&version.to_le_bytes());
    bytes[32..36].copy_from_slice(&salt.to_le_bytes());
    bytes[36..40].copy_from_slice(&domain.to_le_bytes());
    let mut h = Blake2b::<U8>::new();
    h.update(bytes);
    let out = h.finalize();
    u64::from_le_bytes(out.into())
}

fn keyed_uniform(seed: u64, source: u64, pop: u64, version: u64, salt: u32) -> f64 {
    keyed_int(seed, source, pop, version, salt, 1) as f64 / 18446744073709551616.0
}

fn sample_targets(source: usize) -> [usize; DEGREE] {
    let mut out = [0usize; DEGREE];
    for (edge, slot) in out.iter_mut().enumerate() {
        *slot = (keyed_int(SEED, source as u64, 0, 1, edge as u32, 0) % N as u64) as usize;
    }
    out
}

fn build_drive() -> Vec<Vec<usize>> {
    let half = N / 2;
    let mut active_block: HashMap<(usize, usize), bool> = HashMap::new();
    let mut drive = Vec::with_capacity(TICKS);
    for t in 0..TICKS {
        let third = (t * 3 / TICKS).min(2);
        let want_half = if third == 1 { 1 } else { 0 };
        let block = t / DRIVE_BLOCK;
        let mut ids = Vec::new();
        for nid in 0..N {
            let half_of = if nid < half { 0 } else { 1 };
            if half_of != want_half {
                continue;
            }
            let key = (nid, block);
            let active = *active_block.entry(key).or_insert_with(|| {
                keyed_uniform(SEED, nid as u64, DRIVE_POP, 1, block as u32) < DENSITY
            });
            if active {
                ids.push(nid);
            }
        }
        drive.push(ids);
    }
    drive
}

fn run_once(adjacency: &[Vec<usize>], drive: &[Vec<usize>]) -> (u64, f64, f64) {
    let decay = (-DT / TAU).exp();
    let mut v = vec![0.0f64; N];
    let mut refr = vec![0i64; N];
    let mut ring: Vec<Vec<(usize, f64)>> = vec![Vec::new(); DELAY + 1];
    let mut spikes = 0u64;
    let start = Instant::now();
    for t in 0..TICKS {
        let mut inp = vec![0.0f64; N];
        for (tgt, w) in ring[t % (DELAY + 1)].drain(..) {
            inp[tgt] += w;
        }
        for nid in &drive[t] {
            inp[*nid] += AMPLITUDE;
        }
        for nid in 0..N {
            if t as i64 >= refr[nid] {
                v[nid] = inp[nid] + (v[nid] - inp[nid]) * decay;
                if v[nid] >= THRESHOLD {
                    v[nid] = V_RESET;
                    refr[nid] = t as i64 + 1 + REF_TICKS;
                    spikes += 1;
                    let slot = (t + DELAY) % (DELAY + 1);
                    for tgt in &adjacency[nid] {
                        ring[slot].push((*tgt, WEIGHT));
                    }
                }
            } else {
                v[nid] = V_RESET;
            }
        }
    }
    (spikes, v.iter().sum(), start.elapsed().as_secs_f64())
}

fn main() {
    let adjacency: Vec<Vec<usize>> = (0..N).map(|nid| sample_targets(nid).to_vec()).collect();
    let drive = build_drive();
    let (s1, sum1, ms1) = run_once(&adjacency, &drive);
    let (s2, sum2, ms2) = run_once(&adjacency, &drive);
    assert_eq!(
        (s1, sum1.to_bits()),
        (s2, sum2.to_bits()),
        "non-deterministic run"
    );
    println!("spec: N={N} ticks={TICKS} degree={DEGREE} seed={SEED}");
    println!("total_spikes={s1}");
    println!("final_v_sum={sum1:.17e}");
    println!("wall_ms={:.1} / {:.1}", ms1 * 1000.0, ms2 * 1000.0);
}
