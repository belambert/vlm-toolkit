import os
from io import BytesIO
from pathlib import Path

import torch
import typer
from diffusers import DiffusionPipeline
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

app = typer.Typer()

from imggen.util import get_device

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_google_drive_service(credentials_path: str, token_path: str):
    """Authenticate and return Google Drive service."""
    creds = None

    # Check if token.json exists
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh expired token
            creds.refresh(Request())
        else:
            # Run OAuth flow
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    credentials: str = typer.Option("credentials.json", help="Path to OAuth credentials JSON"),
    token: str = typer.Option("token.json", help="Path to save/load OAuth token"),
    filename: str = typer.Option("output.png", help="Filename to save in Google Drive"),
    folder_id: str = typer.Option(None, help="Google Drive folder ID (optional)"),
    num_inference_steps: int = typer.Option(25, help="Number of denoising steps"),
):

    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        # "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(get_device())

    image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Authenticate with Google Drive using OAuth
    service = get_google_drive_service(credentials, token)

    # Prepare file metadata
    file_metadata = {"name": filename}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    # Upload to Google Drive
    media = MediaIoBaseUpload(buffer, mimetype="image/png", resumable=True)
    file = (
        service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id,webViewLink",
        )
        .execute()
    )

    print(f"File uploaded to Google Drive: {file.get('webViewLink')}")
    print(f"File ID: {file.get('id')}")

if __name__ == "__main__":
    app()