import uuid
from datetime import datetime
from io import BytesIO

from google.cloud import storage


def upload_image_to_gcs(
    image_buffer: BytesIO,
    bucket_name: str = "imggen",
    prefix: str = "new",
) -> dict:
    """
    Upload an image to Google Cloud Storage.

    Args:
        image_buffer: BytesIO buffer containing the image data
        bucket_name: Name of the GCS bucket
        prefix: Prefix/folder path in the bucket

    Returns:
        dict with keys: date_folder, filename, public_url, blob_path
    """
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get current date for folder structure
    date_folder = datetime.now().strftime("%Y-%m-%d")

    # Generate unique filename using timestamp and UUID
    timestamp = datetime.now().strftime("%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"image_{timestamp}_{unique_id}.png"

    # Construct full blob path: prefix/date/filename
    blob_path = f"{prefix}/{date_folder}/{filename}"
    blob = bucket.blob(blob_path)

    # Upload to GCS
    image_buffer.seek(0)
    blob.upload_from_file(image_buffer, content_type="image/png")

    # Generate public URL
    public_url = f"gs://{bucket_name}/{blob_path}"

    return {
        "date_folder": date_folder,
        "filename": filename,
        "public_url": public_url,
        "blob_path": blob_path,
    }
