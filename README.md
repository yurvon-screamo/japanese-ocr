# Japanese OCR

> WASM Support

High-performance OCR for recognizing Japanese text from japanese images, written in Rust.

## Description

Japanese OCR is a tool for optical character recognition of Japanese text, optimized for japanese content. The project uses the VisionEncoderDecoderModel architecture with ONNX models for efficient inference on CPU.

### Model Architecture

- **Encoder**: ViT (Vision Transformer) based on `facebook/deit-tiny-patch16-224`
- **Decoder**: BERT based on `tohoku-nlp/bert-base-japanese-char-v2`
- **Format**: ONNX for cross-platform compatibility

## Features

- Recognition of Japanese text from images
- Two operating modes: file and clipboard monitoring
- Automatic copying of the result to the clipboard
- CPU inference without external dependencies

## Installation

### CLI

```bash
cargo install --git https://github.com/yurvon-screamo/japanese-ocr japanese-ocr
```

### Library

```bash
cargo add --git https://github.com/yurvon-screamo/japanese-ocr rs-japanese-ocr
```

## Usage

### Clipboard Mode (Default)

The program monitors the clipboard and automatically recognizes text from appearing images:

```bash
japanese-ocr
```

The recognition result is automatically copied back to the clipboard.

### File Mode

```bash
japanese-ocr --mode file --image path/to/image.png
```

### Command Line Arguments

| Argument                      | Description                          | Default      |
| ----------------------------- | ------------------------------------ | ------------ |
| `-i, --image <PATH>`          | Path to the image for recognition    | —            |
| `--mode <MODE>`               | Operating mode: `file` or `clipboard`| `clipboard`  |
| `--refresh-timeout <SECONDS>` | Clipboard polling interval           | `1.0`        |

## Project Structure

```tree
japanese-ocr/
├── japanese-ocr/           # OCR Library
│   ├── src/
│   │   ├── lib.rs          # Public API
│   │   ├── model.rs        # Model Implementation
│   │   └── error.rs        # Error Handling
│   └── model/              # ONNX models and tokenizer
│       ├── encoder_model.onnx
│       ├── decoder_model.onnx
│       └── tokenizer.json
├── japanese-ocr-cli/       # CLI Application
│   └── src/
│       ├── main.rs         # Entry Point
│       ├── clipboard.rs    # Clipboard Operations
│       └── error.rs        # Error Handling
└── Cargo.toml              # Workspace Configuration
```

## Technical Details

### Image Preprocessing

- Input image size: 224×224 pixels
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- Resize method: Nearest Neighbor

### Text Generation

- Maximum sequence length: 300 tokens
- Autocorrection with `[CLS]` (start) and `[SEP]` (end) tokens
- Removal of spaces from the final result

### Dependencies

| Library       | Purpose                          |
| ------------- | -------------------------------- |
| `candle-core` | Tensor computations              |
| `candle-onnx` | Working with ONNX models         |
| `tokenizers`  | Text tokenization (HuggingFace)  |
| `image`       | Image processing                 |
| `clap`        | Parsing CLI arguments            |
| `arboard`     | Clipboard operations             |

## Usage as a Library

```rust
use rs_japanese_ocr::JapaneseOCRModel;
use image;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = JapaneseOCRModel::load()?;
    let img = image::ImageReader::open("japanese.png")?.decode()?;
    let text = model.run(&img)?;
    println!("Recognized text: {}", text);
    Ok(())
}
```

Add to `Cargo.toml`:

```toml
[dependencies]
japanese-ocr = { path = "path/to/japanese-ocr" }
image = "0.25"
```

## License

The project is distributed under the GNU AGPL v3 license. Details in the [LICENSE](LICENSE) file.
