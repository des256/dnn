# CAT/DOG CLASSIFICATION

This is the first full pipeline attempt. The DNN is a simple binary image classifier for the Microsoft cat/dog dataset.

## Howto

### Training

Train with Keras (TensorFlow), quantize the model into a Tensorflow Lite model, and convert it to ONNX. Use CUDA if/where possible. The objective is to play around with the tools and hyperparameters. Test with different output formats, like int8, float16, bfloat16, etc.

- Go to `train/`.
- Download the dataset with `fetch_data.py`. Unzip the downloaded file and make sure there is a new `PetImages` directory now.
- Run `train/main.py`, this will clean the dataset, train the model, quantize it, and save as `catdog.tflite`.
- Run `tflite2onnx.sh catdog.tflite ../catdog.onnx` to convert the tflite model to onnx and put it in the root directory.

### Inference

Figure out some simple way to run inference from a Rust program. Use CUDA if/where possible. Attempt to run on Pi5 and Jetson.

- `cargo build` the Rust code.
- download a few images of cats and dogs from the internet and put them in `testdata/`.
- run `target/debug/catdog`. This will load the onnx model and run inference on the images.

## Experiences

Keras is very easy to use. There is very little boilerplate code. Specifying the model is very intuitive. It takes a bit of time to get to understand the jargon, but this will probably improve by doing more examples. Quantization is straightforward by just using quantization-aware training. Image manipulation is easier than it used to be. I like the functionality to augment the input data during training. This was always a bit of a hassle in the past. Further down the road it might be possible to run zero-copy inference directly from camera on Jetson or Pi5. The model has 2.9M parameters. Qunatization-aware training is slower and slightly less accurate depending on the data formats, but the ease of use for the entire pipeline is worth the wait.

## Further Experiments

### Epochs

A quick test was done with 5 epochs and the results looked promising (as in, it seems that this will work). Now let's do a real run with 50 epochs to get the loss down to somewhat of a minimum. This takes about an hour on my rig. The resulting model is 96% accurate on the validation set. The resulting ONNX inference is 730us (CUDA) or 3600us (CPU) per image on my rig.

### Raspberry Pi 5

It runs on CPU at around 38ms per image. Would be interesting to see if there is some implementation that uses Vulkan compute shaders... Maybe wonnx.