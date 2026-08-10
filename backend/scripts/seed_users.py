"""
Seed the database with default users for Phase 4.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db, auth

def seed():
    db.init()
    
    users_to_create = [
        ("admin1", "password123", "IT Admin"),
        ("reviewer1", "password123", "Reviewer"),
        ("translator1", "password123", "Legal Translator"),
        ("officer1", "password123", "Desk Officer")
    ]
    
    print("Seeding users...")
    for username, password, role in users_to_create:
        if not db.get_user_by_username(username):
            hashed = auth.get_password_hash(password)
            db.create_user(username, hashed, role)
            print(f"Created {role}: {username}")
        else:
            print(f"User {username} already exists.")
            
if __name__ == "__main__":
    seed()
