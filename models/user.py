from flask_login import UserMixin
from werkzeug.security import generate_password_hash
import uuid

class User(UserMixin):
    def __init__(self, id, email, username, push_token=None):
        self.id = id
        self.email = email
        self.username = username
        self.push_token = push_token
    
    @staticmethod
    def get(user_id, users_ref):
        doc = users_ref.document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            return User(doc.id, data['email'], data['name'], data.get('push_token'))
        return None

    @staticmethod
    def get_by_email(email, db):
        if not db: return None
        docs = db.collection('users').where('email', '==', email).limit(1).stream()
        for doc in docs:
            data = doc.to_dict()
            return User(doc.id, data.get('email'), data.get('name'), data.get('push_token'))
        return None

    @staticmethod
    def create(email, name, password, db):
        if not db:
            print("ERROR: Database not initialized in User.create()")
            return None
        try:
            user_id = str(uuid.uuid4())
            password_hash = generate_password_hash(password)
            db.collection('users').document(user_id).set({
                'email': email,
                'name': name,
                'password_hash': password_hash,
                'created_at': None,
                'push_token': None,
                'daily_reminder': False
            })
            print(f"User created successfully: {user_id} ({email})")
            return User(user_id, email, name)
        except Exception as e:
            print(f"ERROR creating user: {e}")
            return None

class AnonymousUser(UserMixin):
    def get_id(self):
        from flask import session
        if "anonymous_id" not in session:
            session["anonymous_id"] = str(uuid.uuid4())
        return session.get("anonymous_id")

    @property
    def is_authenticated(self):
        return False
