from source.helpers.validation import *
from source import app
from pathlib import Path
import sys



def main() -> None:
    
    cli_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    image_path = resolve_image_target(cli_path)

    if image_path is None:
        folder_notexist()
        return

    print(f"Image resolved for processing: {image_path}")
    app.process_image(image_path)


if __name__ == '__main__':
    main()
