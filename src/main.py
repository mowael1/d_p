import os

import uvicorn

from src.helpers.config import get_settings

settings = get_settings()


def main():

    port = int(os.environ.get("PORT", settings.API_PORT))

    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=port,
    )


if __name__ == "__main__":
    main()