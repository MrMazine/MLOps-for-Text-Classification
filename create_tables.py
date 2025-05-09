from app.utils.database import db
from app import create_app

# Create the Flask app
app = create_app()

# Create all tables
with app.app_context():
    db.create_all()
    print("All tables created successfully.")
