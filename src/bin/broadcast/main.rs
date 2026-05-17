use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const WIDTH: usize = 4;
const HEIGHT: usize = 4;

#[cuda_module]
mod kernels {

    use super::{DisjointSlice, WIDTH, kernel, thread};

    #[kernel]
    pub extern "Rust" fn map(
        a: &[f32],
        height: usize,
        mut output: DisjointSlice<f32, thread::Index2D<WIDTH>>,
    ) {
        let row = thread::index_2d_row();

        if let Some(idx) = thread::index_2d::<WIDTH>()
            && row < height
        {
            let i = idx.get();
            if let Some(out) = output.get_mut(idx) {
                *out = a[i] + 100.0;
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let a_host = vec![0f32; WIDTH * HEIGHT];

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let mut output_dev = DeviceBuffer::zeroed(&stream, WIDTH * HEIGHT)?;

    kernels::load(&ctx)?.map(
        &stream,
        LaunchConfig {
            grid_dim: (1, u32::try_from(HEIGHT).unwrap(), 1),
            block_dim: (u32::try_from(WIDTH).unwrap(), 1, 1),
            shared_mem_bytes: 0,
        },
        &a_dev,
        2,
        &mut output_dev,
    )?;

    let out_host = output_dev.to_host_vec(&stream)?;

    assert!(!out_host.is_empty());

    assert_eq!(out_host, [[100.0; 8], [0.0; 8]].concat().clone());

    println!("Woot");

    Ok(())
}

#[allow(dead_code)]
fn next_u64(seed: u64) -> u64 {
    // Increment the state by the Golden Ratio constant
    let state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);

    // Apply mixing functions
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bla() {
        dbg!(next_u64(42));
        assert!(false);
    }
}
