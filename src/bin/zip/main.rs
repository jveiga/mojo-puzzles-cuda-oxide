use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: u16 = 1024;

#[cuda_module]
mod kernels {
    use super::{DisjointSlice, kernel, thread};

    #[kernel]
    pub extern "Rust" fn map(a: &[f32], b: &[f32], mut output: DisjointSlice<f32>) {
        if let Some(out) = output.get_mut(thread::index_1d()) {
            *out = a[thread::index_1d().get() as usize] + b[thread::index_1d().get() as usize];
        }
    }
}

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let a_host: Vec<_> = (0..N).map(f32::from).collect();
    let b_host: Vec<f32> = (0..N).map(f32::from).map(|i| 10.0f32 + i).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut output_dev = DeviceBuffer::zeroed(&stream, usize::from(N))?;

    kernels::load(&ctx)?.map(
        &stream,
        LaunchConfig::for_num_elems(u32::from(N)),
        &a_dev,
        &b_dev,
        &mut output_dev,
    )?;

    let out_host = output_dev.to_host_vec(&stream)?;

    for (idx, n) in out_host.into_iter().enumerate() {
        assert!((a_host[idx] + b_host[idx] - n).abs() < 0.1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_fails() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        const N: usize = 1024;
        let a_host: Vec<_> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<_> = (0..N).map(|i| 10.0f32 + i as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut output_dev = DeviceBuffer::zeroed(&stream, N).unwrap();

        kernels::load(&ctx)
            .unwrap()
            .map(
                &stream,
                LaunchConfig::for_num_elems(u32::try_from(N).unwrap()),
                &a_dev,
                &b_dev,
                &mut output_dev,
            )
            .unwrap();

        let out_host = output_dev.to_host_vec(&stream).unwrap();

        for (idx, n) in out_host.into_iter().enumerate() {
            assert_eq!(a_host[idx] + b_host[idx], n);
        }
    }
}
