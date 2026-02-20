use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Library error: {0}")]
    Lib(#[from] rs_manga_ocr::MangaOCRError),

    #[error("Clipboard error: {0}")]
    Clipboard(#[from] arboard::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
