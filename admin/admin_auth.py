from werkzeug.security import check_password_hash

ADMIN_EMAIL = "admin@sahaay.ai"
# Password: admin123 (change this hash if you want a different password)
ADMIN_PASSWORD_HASH = "scrypt:32768:8:1$GbTFt1JXzOhCNzob$f62a2f31998bbddb7e39dd99a0190dfe79481d101f5c739704f5fcb0fdfba50b3b24c9967f02847c586d278584e9b887afa29a377e47c00f46729b055122366e"

def validate_admin(email, password):
    return email == ADMIN_EMAIL and check_password_hash(ADMIN_PASSWORD_HASH, password)
