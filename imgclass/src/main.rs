use {
    image::{imageops::FilterType, GenericImageView},
    ndarray::Array,
    std::{
        collections::HashMap,
        time::{Duration, Instant},
    },
    wonnx::{
        utils::{InputTensor, OutputTensor},
        Session,
    },
};

const ONNX_PATH: &str = "catdog.onnx";
const REPEATS: usize = 50;

async fn check_image(model: &Session, path: &str) -> (f32, Duration) {
    // load image
    println!("    loading image: {}...",path);
    let img = image::open(path).unwrap();

    // find shortest side
    let (width, height) = img.dimensions();

    // crop to square
    println!("    cropping to square...");
    let img = if width < height {
        img.crop_imm(0, (height - width) / 2, width, width)
    } else {
        img.crop_imm((width - height) / 2, 0, height, height)
    };

    // resize to 180x180
    println!("    resizing to 180x180...");
    let img = img.resize_exact(180, 180, FilterType::Triangle);

    // build float array from the image pixels
    println!("    building float array...");
    let mut input = Array::zeros([1, 180, 180, 3]);
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, y, x, 0]] = r as f32 / 255.0;
        input[[0, y, x, 1]] = g as f32 / 255.0;
        input[[0, y, x, 2]] = b as f32 / 255.0;
    }

    let mut input_data = HashMap::<String, InputTensor>::new();
    input_data.insert(
        "serving_default_input_1:0".to_string(),
        input.as_slice().unwrap().into(),
    );

    // run inference
    println!("    running and measuring inference...");
    let start = Instant::now();
    let mut x = 0.0f32;
    for _ in 0..REPEATS {
        let outputs = model.run(&input_data).await.unwrap();
        let output = outputs.get("StatefulPartitionedCall:0").unwrap();
        x = match output {
            OutputTensor::U8(data) => (data[0] as f32) / 255.0,
            OutputTensor::F32(data) => data[0],
            OutputTensor::I32(data) => data[0] as f32,
            OutputTensor::I64(data) => data[0] as f32,
        };
    }
    let end = Instant::now();

    // return sigmoid and inference duration
    (
        1.0 / (1.0 + (-x).exp()),
        end.duration_since(start).div_f32(REPEATS as f32),
    )
}

async fn async_main() {
    // load the model
    let model = Session::from_path(ONNX_PATH).await.unwrap();

    // run inference on all images in the testdata directory
    println!("running inference:");
    let image_paths = std::fs::read_dir("testdata").unwrap();
    for image_path in image_paths {
        let image_path = image_path.unwrap().path();
        let image_name = image_path.file_name().unwrap().to_str().unwrap();
        let (prediction, duration) = check_image(&model, image_path.to_str().unwrap()).await;
        if prediction > 0.5 {
            println!(
                "    {} is a dog ({:.2}us)",
                image_name,
                duration.as_micros() as f32,
            );
        } else {
            println!(
                "    {} is a cat ({:.2}us)",
                image_name,
                duration.as_micros() as f32,
            );
        }
    }
}

fn main() {
    pollster::block_on(async_main());
}
