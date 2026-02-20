# Manga OCR

> WASM Support

High-performance OCR for recognizing Japanese text from manga images, written in Rust.

## Description

Manga OCR is a tool for optical character recognition of Japanese text, optimized for manga content. The project uses the VisionEncoderDecoderModel architecture with ONNX models for efficient inference on CPU.

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
cargo install manga-ocr
```

### Library

```bash
cargo add rs-manga-ocr
```

## Usage

### Clipboard Mode (Default)

The program monitors the clipboard and automatically recognizes text from appearing images:

```bash
manga-ocr
```

The recognition result is automatically copied back to the clipboard.

### File Mode

```bash
manga-ocr --mode file --image path/to/image.png
```

### Command Line Arguments

| Argument                      | Description                          | Default      |
| ----------------------------- | ------------------------------------ | ------------ |
| `-i, --image <PATH>`          | Path to the image for recognition    | —            |
| `--mode <MODE>`               | Operating mode: `file` or `clipboard`| `clipboard`  |
| `--refresh-timeout <SECONDS>` | Clipboard polling interval           | `1.0`        |

## Project Structure

```
manga-ocr/
├── rs-manga-ocr/           # OCR Library
│   ├── src/
│   │   ├── lib.rs          # Public API
│   │   ├── model.rs        # Model Implementation
│   │   └── error.rs        # Error Handling
│   └── model/              # ONNX models and tokenizer
│       ├── encoder_model.onnx
│       ├── decoder_model.onnx
│       └── tokenizer.json
├── rs-manga-ocr-cli/       # CLI Application
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
use rs_manga_ocr::MangaOCRModel;
use image;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = MangaOCRModel::load()?;
    let img = image::ImageReader::open("manga.png")?.decode()?;
    let text = model.run(&img)?;
    println!("Recognized text: {}", text);
    Ok(())
}
```

Add to `Cargo.toml`:

```toml
[dependencies]
rs-manga-ocr = { path = "path/to/rs-manga-ocr" }
image = "0.25"
```

## License

The project is distributed under the GNU AGPL v3 license. Details in the [LICENSE](LICENSE) file.
