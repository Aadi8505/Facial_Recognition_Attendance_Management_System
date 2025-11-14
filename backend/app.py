from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime, date
from deepface import DeepFace
import cv2
import numpy as np
from numpy.linalg import norm


app = Flask(__name__)


# --- Database configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "attendance.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'connect_args': {'timeout': 20}
}


db = SQLAlchemy(app)
with app.app_context():
    with db.engine.connect() as conn:
        conn.execute(db.text("PRAGMA journal_mode=WAL;"))


# --- Models ---
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # admin-provided numeric ID
    name = db.Column(db.String(100), nullable=False)
    course = db.Column(db.String(100))
    section = db.Column(db.String(50))

class Teacher(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # admin-provided numeric ID
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(100))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)   # login id (numeric) - same as Student.id or Teacher.id for those roles
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'student', 'teacher', 'admin'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SectionStudent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    section = db.Column(db.String(50), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)

class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject_name = db.Column(db.String(100), nullable=False)
    section = db.Column(db.String(50), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id'), nullable=False)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id'), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(10), nullable=False)  # "Present" or "Absent"

class FacialEncoding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    encoding = db.Column(db.PickleType, nullable=False)  # stores face vector
    date_added = db.Column(db.DateTime, default=datetime.utcnow)

# --- Routes ---
@app.route("/")
def home():
    return jsonify({"message": "Attendance system backend connected successfully!"})

# Admin-only: Add user (student/teacher/admin)
# Required JSON for students: { "id": 2311981003, "name":"Aaditya", "password":"pwd", "role":"student", "course":"CSE", "section":"A" }
# For teacher: { "id": 5001, "name":"Raj", "password":"pwd", "role":"teacher", "subject":"DSA" }
# For admin: { "id": 1, "name":"admin", "password":"pwd", "role":"admin" }
@app.route("/add_user", methods=["POST"])
def add_user():
    data = request.get_json()
    try:
        provided_id = int(data["id"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Field 'id' (numeric) is required."}), 400

    name = data.get("name", "")
    password = data.get("password", "")
    role = data.get("role", "").lower()

    if not password or role not in ("student", "teacher", "admin"):
        return jsonify({"error": "Fields 'password' and valid 'role' are required."}), 400

    # Prevent duplicate user id
    if User.query.get(provided_id):
        return jsonify({"error": f"User id {provided_id} already exists."}), 400

    # Create role-specific record (for student/teacher)
    if role == "student":
        # prevent duplicate student id
        if Student.query.get(provided_id):
            return jsonify({"error": f"Student id {provided_id} already exists in Student table."}), 400
        new_student = Student(
            id=provided_id,
            name=name or f"Student_{provided_id}",
            course=data.get("course", ""),
            section=data.get("section", "")
        )
        db.session.add(new_student)
        db.session.flush()
        # Add section mapping
        section_map = SectionStudent(
            section=new_student.section,
            student_id=new_student.id
        )
        db.session.add(section_map)

    elif role == "teacher":
        if Teacher.query.get(provided_id):
            return jsonify({"error": f"Teacher id {provided_id} already exists in Teacher table."}), 400
        new_teacher = Teacher(
            id=provided_id,
            name=name or f"Teacher_{provided_id}",
            subject=data.get("subject", "")
        )
        db.session.add(new_teacher)
        db.session.flush()

    # Admins don't need a Student/Teacher entry

    # Create User entry (login)
    user = User(
        id=provided_id,
        role=role
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": f"{role.capitalize()} with id {provided_id} added successfully."})

# Login route (for all roles) - uses numeric id
# JSON: { "id": 2311981003, "password": "pwd" }
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    try:
        provided_id = int(data["id"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Field 'id' (numeric) is required."}), 400

    password = data.get("password", "")
    if not password:
        return jsonify({"error": "Password is required."}), 400

    user = User.query.get(provided_id)
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid id or password"}), 401

    return jsonify({
        "message": f"Login successful as {user.role}",
        "role": user.role,
        "id": user.id
    })

@app.route("/add_subject", methods=["POST"])
def add_subject():
    data = request.get_json()

    try:
        subject_id = int(data["id"])
        subject_name = data["subject_name"]
        section = data["section"]
        teacher_id = int(data["teacher_id"])
    except:
        return jsonify({"error": "id, subject_name, section, teacher_id are required"}), 400

    teacher = Teacher.query.get(teacher_id)
    if not teacher:
        return jsonify({"error": "Teacher ID not found"}), 404

    subject = Subject(
        id=subject_id,
        subject_name=subject_name,
        section=section,
        teacher_id=teacher_id
    )
    db.session.add(subject)
    db.session.commit()

    return jsonify({"message": "Subject added successfully!"})

@app.route("/add_face_encoding", methods=["POST"])
def add_face_encoding():
    try:
        student_id = int(request.form.get("student_id"))
    except:
        return jsonify({"error": "student_id must be numeric"}), 400

    # 1. Check student exists
    student = Student.query.get(student_id)
    if not student:
        return jsonify({"error": f"Student with id {student_id} does not exist"}), 404

    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    image_file = request.files["image"]

    # Read image
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image or unable to decode"}), 400

    # 2. Perform encoding BEFORE any DB operations
    try:
        embedding_result = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            detector_backend="mtcnn"
        )
        encoding = embedding_result[0]["embedding"]

    except Exception as e:
        return jsonify({"error": "Face could not be encoded", "details": str(e)}), 400

    # 3. DB write AFTER slow operation is done
    try:
        face_record = FacialEncoding(
            student_id=student_id,
            encoding=encoding
        )
        db.session.add(face_record)
        db.session.commit()

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "DB error", "details": str(e)}), 500

    return jsonify({"message": "Face encoding stored successfully"}), 200

# recognise and mark
SIMILARITY_THRESHOLD = 0.50

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.all(a == 0) or np.all(b == 0):
        return -1.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

@app.route("/recognize_and_mark", methods=["POST"])
def recognize_and_mark():
    # 1) Validate inputs
    if "image" not in request.files:
        return jsonify({"error": "image file is required (form-data key: image)"}), 400

    try:
        subject_id = int(request.form.get("subject_id"))

    except (TypeError, ValueError):
        return jsonify({"error": "subject_id and teacher_id are required and must be numeric"}), 400

    # Check subject/teacher exist
    subject = Subject.query.get(subject_id)
    if not subject:
        return jsonify({"error": f"Subject id {subject_id} not found"}), 404

    teacher_id = subject.teacher_id
    teacher = Teacher.query.get(teacher_id)

    image_file = request.files["image"]
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image or unable to decode"}), 400

    # 2) Extract embedding BEFORE touching DB (slow op first)
    try:
        rep = DeepFace.represent(img_path=img, model_name="ArcFace", detector_backend="mtcnn")
        probe_embedding = rep[0]["embedding"]
    except Exception as e:
        return jsonify({"error": "Face encoding failed", "details": str(e)}), 400

    # 3) Load known encodings from DB
    # We will build a mapping: student_id -> list of embeddings
    enc_rows = FacialEncoding.query.all()
    if not enc_rows:
        return jsonify({"error": "No face encodings in database yet"}), 400

    student_encs = {}
    for r in enc_rows:
        sid = r.student_id
        # ensure embedding is array-like
        emb = r.encoding
        student_encs.setdefault(sid, []).append(np.array(emb, dtype=np.float32))

    # 4) Compute best match per student (max similarity among their encodings)
    best_student = None
    best_score = -1.0
    for sid, embeddings in student_encs.items():
        # compute max similarity for this student
        scores = [cosine_similarity(probe_embedding, e) for e in embeddings]
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_student = sid

    # 5) Decide match or unknown
    if best_score < SIMILARITY_THRESHOLD or best_student is None:
        return jsonify({
            "result": "unknown",
            "message": "No matching student found",
            "best_score": best_score
        }), 200

    # 6) Prevent double-marking for same student-subject-date
    today_str = date.today().isoformat()
    existing = Attendance.query.filter_by(
        student_id=best_student,
        subject_id=subject_id,
        date=today_str
    ).first()

    if existing:
        return jsonify({
            "result": "already_marked",
            "student_id": best_student,
            "score": best_score,
            "message": "Attendance already marked for today"
        }), 200

    # 7) Insert attendance entry ( Present )
    try:
        att = Attendance(
            student_id=best_student,
            subject_id=subject_id,
            teacher_id=teacher_id,
            date=today_str,
            status="Present"
        )
        db.session.add(att)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "DB write failed", "details": str(e)}), 500

    # 8) Return success with student details
    student_obj = Student.query.get(best_student)
    return jsonify({
        "result": "present",
        "student_id": best_student,
        "student_name": student_obj.name if student_obj else None,
        "score": best_score,
        "subject_id": subject_id,
        "date": today_str
    }), 200

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

# git add .
# git commit -m "Made the facial recognition model working, which marks attendance of students in attendance table, frontend not displaying attendance marked message"
# git push

