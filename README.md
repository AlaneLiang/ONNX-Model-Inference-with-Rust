# ONNX Model Inference with Rust

This Rust program demonstrates how to perform inference using an ONNX model for text classification.

## Prerequisites

Make sure you have Rust installed on your machine. You can install it by following the instructions on the [official Rust website](https://www.rust-lang.org/).

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/AlaneLiang/ONNX-Model-Inference-with-Rust.git
    ```

2. Change into the project directory:

    ```bash
    cd ONNX-Model-Inference-with-Rust

    ```

3. Build and run the program:

    ```bash
    cargo run
    ```

## Usage

You can modify the `dict.json` file with your own word-to-index mapping and replace the `model.onnx` file with your trained ONNX model. And you need to change the input vector shape according to your model input size.

## Folder Structure

- `src`: Contains the Rust source code files.
- `dict.json`: JSON file containing the word-to-index mapping.
- `model.onnx`: ONNX model file for text classification.

## Acknowledgments

- [Open Neural Network Exchange (ONNX)](https://onnx.ai/): Open format for AI models.
- [ndarray](https://docs.rs/ndarray/): N-dimensional arrays.
- [serde](https://serde.rs/): Serialization and deserialization framework for Rust.

## License

This project is licensed under the MIT License - see the [LICENSE](https://chat.openai.com/c/LICENSE) file for details.
