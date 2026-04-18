from pathlib import Path


INPUT_FOLDER = Path("uploads")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_supported_image(image_path: Path) -> bool:
    return image_path.suffix.lower() in SUPPORTED_EXTENSIONS


def validate_image_path(image_path: Path) -> bool:
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return False

    if not image_path.is_file():
        print(f"Path is not a file: {image_path}")
        return False

    if not is_supported_image(image_path):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"Unsupported image type '{image_path.suffix}'. Supported types: {supported}")
        return False

    return True


def find_valid_image_in_folder(folder: Path) -> Path | None:
    if not folder.exists() or not folder.is_dir():
        return None

    for file_path in folder.iterdir():
        if validate_image_path(file_path):
            return file_path

    return None


def resolve_image_target(cli_path: Path | None = None) -> Path | None:
    if cli_path is not None:
        return cli_path if validate_image_path(cli_path) else None
    return find_valid_image_in_folder(INPUT_FOLDER)


def folder_notexist() -> None:
    print("No image path provided and no valid image found in the uploads folder.")
    print(f"Please pass a valid image path or add a supported image to '{INPUT_FOLDER}'.")
