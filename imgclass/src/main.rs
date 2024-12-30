use {
    image::{imageops::FilterType, GenericImageView},
    ndarray::Array,
    ort::{
        execution_providers::{ACLExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider},
        inputs,
        session::Session,
    },
    std::time::{Duration, Instant},
};

const ONNX_PATH: &str = "catdog.onnx";
const REPEATS: usize = 50;

fn check_image(model: &Session, path: &str) -> (f32, Duration) {
    // load image
    println!("    loading image: {}...", path);
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

    // run inference
    println!("    running and measuring inference...");
    let start = Instant::now();
    let mut x = 0.0f32;
    for _ in 0..REPEATS {
        let outputs = model.run(inputs![input.clone()].unwrap()).unwrap();
        let output = outputs["dense"].try_extract_tensor::<f32>().unwrap();
        x = output.as_slice().unwrap()[0];
    }
    let end = Instant::now();

    // return sigmoid and inference duration
    (
        1.0 / (1.0 + (-x).exp()),
        end.duration_since(start).div_f32(REPEATS as f32),
    )
}

fn main() {
    // initialize ONNX session
    ort::init()
        //.with_execution_providers([CPUExecutionProvider::default().build().error_on_failure()])
        .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
        .commit()
        .unwrap();

    // load the model
    let model = Session::builder()
        .unwrap()
        .commit_from_file(ONNX_PATH)
        .unwrap();

    // run inference on all images in the testdata directory
    println!("running inference:");
    let image_paths = std::fs::read_dir("testdata").unwrap();
    for image_path in image_paths {
        let image_path = image_path.unwrap().path();
        let image_name = image_path.file_name().unwrap().to_str().unwrap();
        let (prediction, duration) = check_image(&model, image_path.to_str().unwrap());
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
