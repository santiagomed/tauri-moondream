use candle::{utils, Device, Result, Tensor};

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 378, 378).
pub fn load_image<P: AsRef<std::path::Path>>(p: P) -> Result<Tensor> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle); // Adjusted to 378x378
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (378, 378, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
