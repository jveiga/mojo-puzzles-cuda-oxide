use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: u16 = 1024;

#[cuda_module]
mod kernels {
    use super::{DisjointSlice, kernel, thread};

    #[kernel]
    pub extern "Rust" fn guard(a: &[f32], mut output: DisjointSlice<f32>) {
        if let Some(out) = output.get_mut(thread::index_1d()) {
            *out = a[thread::index_1d().get() as usize];
        }
    }
}

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let a_host: Vec<_> = (0..N).map(f32::from).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let mut output_dev = DeviceBuffer::zeroed(&stream, usize::from(N))?;

    kernels::load(&ctx)?.guard(
        &stream,
        LaunchConfig::for_num_elems(u32::from(N)),
        &a_dev,
        &mut output_dev,
    )?;

    let out_host = output_dev.to_host_vec(&stream)?;

    assert!(!out_host.is_empty());

    for (idx, n) in out_host.into_iter().enumerate() {
        assert!((f32::from(u16::try_from(idx).unwrap()) - n).abs() < 0.1);
    }
    Ok(())
}
