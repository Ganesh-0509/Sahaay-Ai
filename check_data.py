"""
Quick diagnostic script to check Firestore data for user

Run this to see what data exists in your Firestore database
"""

import os
from google.cloud import firestore
from google.oauth2 import service_account
import json

# Load service account
cred_path = 'sahaay-ai-7fd925852862.json'
creds = service_account.Credentials.from_service_account_file(
    cred_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Get project ID
with open(cred_path, 'r') as f:
    project_id = json.load(f).get('project_id')

# Initialize Firestore
db = firestore.Client(project=project_id, credentials=creds)

print("=" * 60)
print("FIRESTORE DATA DIAGNOSTIC")
print("=" * 60)

# Get all users
users_ref = db.collection('users')
users = list(users_ref.stream())

print(f"\nTotal users in database: {len(users)}")
print("-" * 60)

for user_doc in users:
    user_id = user_doc.id
    user_data = user_doc.to_dict()
    
    print(f"\nUser ID: {user_id}")
    print(f"Email: {user_data.get('email', 'N/A')}")
    print(f"Username: {user_data.get('username', 'N/A')}")
    
    # Check checkins
    checkins_ref = users_ref.document(user_id).collection('checkins')
    checkins =list(checkins_ref.stream())
    print(f"Total check-ins: {len(checkins)}")
    
    if checkins:
        print("Recent check-ins:")
        for checkin in checkins[:5]:
            checkin_data = checkin.to_dict()
            doc_id = checkin.id
            actual_date = checkin_data.get('date', 'NO DATE FIELD!')
            # NOTE: save_checkin saves as 'last_text', not 'text'!
            has_text = bool(checkin_data.get('last_text'))
            text_preview = checkin_data.get('last_text', '')[:50] if has_text else 'NO TEXT'
            mood = checkin_data.get('mood_label', checkin_data.get('mood_dominant', 'N/A'))
            
            print(f"  - Doc ID: {doc_id}")
            print(f"    Date field: {actual_date}")
            print(f"    Mood: {mood}")
            print(f"    Has text: {has_text}")
            if has_text:
                print(f"    Text preview: {text_preview}...")
            print()
    
    # Check conversations
    convos_ref = users_ref.document(user_id).collection('conversations')
    convos = list(convos_ref.stream())
    print(f"Total conversations: {len(convos)}")
    
    print("-" * 60)

print("\nDiagnostic complete!")
print("\nLOOK FOR:")
print("1. Your email address in the users list")
print("2. The user_id associated with that email")
print("3. Number of check-ins and conversations for that user")
