import logging
import os
import sys

import uvicorn
from pydantic import ValidationError

from fish_memory.auth.cli import run_auth_command_from_environment
from fish_memory.config import Settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def configure_application_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("fish_memory").setLevel(logging.INFO)


def read_server_host(value: str | None) -> str:
    host = DEFAULT_HOST if value is None else value.strip()
    if not host:
        raise ValueError("FISH_MEMORY_HOST must not be blank")
    return host


def read_server_port(value: str | None) -> int:
    try:
        port = DEFAULT_PORT if value is None else int(value)
    except ValueError as error:
        raise ValueError("FISH_MEMORY_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("FISH_MEMORY_PORT must be between 1 and 65535")
    return port


def format_configuration_error(error: ValidationError) -> str:
    problems: list[str] = []
    for detail in error.errors():
        location = detail.get("loc", ())
        message = str(detail.get("msg", "Invalid value"))
        if location:
            field = str(location[0])
            environment_name = f"FISH_MEMORY_{field.upper()}"
            if detail.get("type") == "missing":
                problems.append(f"- {environment_name} is required.")
            elif detail.get("type") == "string_too_short":
                problems.append(f"- {environment_name} cannot be blank.")
            else:
                problems.append(f"- {environment_name}: {message}")
        else:
            problems.append(f"- {message.removeprefix('Value error, ')}")

    details = "\n".join(problems)
    return (
        "fish-memory cannot start because its configuration is invalid:\n"
        f"{details}\n\n"
        "Set the required values as environment variables or in a .env file\n"
        "in the working directory, for example:\n"
        "  FISH_MEMORY_LLM_API_KEY=...\n"
    )


def validate_startup_configuration() -> None:
    try:
        Settings()
    except ValidationError as error:
        raise SystemExit(format_configuration_error(error)) from None


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "auth":
        raise SystemExit(run_auth_command_from_environment(sys.argv[2:]))
    validate_startup_configuration()
    configure_application_logging()
    uvicorn.run(
        "fish_memory.app:create_app",
        factory=True,
        host=read_server_host(os.getenv("FISH_MEMORY_HOST")),
        port=read_server_port(os.getenv("FISH_MEMORY_PORT")),
        workers=1,
    )


if __name__ == "__main__":
    main()
