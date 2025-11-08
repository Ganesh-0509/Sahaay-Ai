import os
import json
from google.oauth2 import service_account
from google.cloud import firestore


def test_firestore_connection():
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    print(f"GOOGLE_APPLICATION_CREDENTIALS = {cred_path}")

    if not cred_path or not os.path.exists(cred_path):
        print("Credentials file not found or GOOGLE_APPLICATION_CREDENTIALS not set.")
        return False

    # Read and show the project_id inside the JSON to confirm
    try:
        with open(cred_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        print(f"project_id in JSON: {data.get('project_id')}")
    except Exception as e:
        print(f"Failed to read/parse JSON: {e}")
        return False

    # Build credentials explicitly from the file and force the client to use that project
    try:
        creds = service_account.Credentials.from_service_account_file(cred_path)
        # Prefer project from JSON, fall back to common env vars
        project = data.get('project_id') or os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCLOUD_PROJECT') or os.getenv('PROJECT_ID')
        print(f"Attempting to initialize Firestore client with project='{project}'")

        db = firestore.Client(project=project, credentials=creds)
        print(f"firestore.Client().project = {db.project}")

        # Try listing collections to validate access
        collections = [c.id for c in db.collections()]
        print("✅ Successfully connected to Firestore")
        print(f"Available collections: {collections}")
        return True
    except Exception as e:
        print("❌ Error connecting to Firestore:")
        print(f"Error details: {e}")
        return False


if __name__ == '__main__':
    test_firestore_connection()