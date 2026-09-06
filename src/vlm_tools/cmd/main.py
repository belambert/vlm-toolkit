"""The `vlm` entry point, dispatching to the per-command Typer apps."""

import importlib
from typing import cast

import click
import typer

# name -> (module under vlm_tools.cmd, one-line help shown by `vlm --help`)
COMMANDS = {
    "process": ("vlm_process", "Run a VLM over a folder of images locally"),
    "server": ("vlm_server", "Run a folder through an OpenAI-compatible endpoint"),
    "caption-images": ("caption_images", "Caption images with the captioning prompt"),
    "detect-logo": ("detect_logo", "Find logos and watermarks with a VLM"),
    "remove-logo": ("remove_logo", "Erase detected boxes by inpainting"),
    "view-output": ("view_vlm_output", "Render a run as an HTML page"),
    "upload-dataset": ("upload_dataset", "Publish a run to the Hugging Face Hub"),
}

# top-level module -> the extra that provides it, for a readable error when a
# command is run against a base install
EXTRAS = {
    "accelerate": "local",
    "kernels": "local",
    "outlines": "local",
    "torch": "local",
    "torchvision": "local",
    "transformers": "local",
    "cv2": "logo",
    "numpy": "logo",
    "datasets": "hub",
}


class LazyGroup(click.Group):
    """Import a subcommand's module only when that subcommand actually runs.

    Keeps `vlm --help` and the light commands usable without the optional heavy
    extras (torch, opencv, datasets) installed.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(COMMANDS)

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        if name not in COMMANDS:
            return None
        module = importlib.import_module(f"vlm_tools.cmd.{COMMANDS[name][0]}")
        command = typer.main.get_command(module.app)
        command.short_help = COMMANDS[name][1]
        # typer vendors its own click, so its commands aren't click.Command
        # subclasses; click only ever duck-types what it gets back from here
        return cast(click.Command, command)

    def invoke(self, ctx: click.Context) -> object:
        # typer's Exit and Abort are unrelated to click's, so without this a
        # plain click group lets them escape as unhandled exceptions - which
        # turns `--help` and every `raise typer.Exit(1)` into a traceback-ish 1
        try:
            return super().invoke(ctx)
        except typer.Exit as e:
            raise click.exceptions.Exit(e.exit_code) from e
        except typer.Abort as e:
            raise click.exceptions.Abort() from e
        except ModuleNotFoundError as e:
            extra = EXTRAS.get(e.name or "")
            if extra is None:
                raise
            raise click.ClickException(
                f"`vlm {ctx.invoked_subcommand}` needs the '{extra}' extra.\n"
                f"Install it with: pip install 'vlm-toolkit[{extra}]'"
            ) from e

    def format_commands(self, ctx: click.Context, formatter) -> None:  # type: ignore[no-untyped-def]
        # use the static help text, so listing commands imports nothing
        with formatter.section("Commands"):
            formatter.write_dl([(name, help) for name, (_, help) in COMMANDS.items()])


@click.group(cls=LazyGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="vlm-toolkit")
def cli() -> None:
    """Batch image processing with vision language models."""


if __name__ == "__main__":
    cli()
