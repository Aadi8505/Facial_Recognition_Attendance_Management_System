from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime, date
from deepface import DeepFace
import cv2
import numpy as np
from numpy.linalg import norm
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

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

    # --- Extract required fields ---
    try:
        custom_id = int(data["id"])  # YOU provide the ID
    except:
        return jsonify({"error": "id must be numeric"}), 400

    name = data.get("name")
    password = data.get("password")
    role = data.get("role")

    if not all([name, password, role]):
        return jsonify({"error": "name, password, and role are required"}), 400

    # --- Check if ID already exists in User table ---
    if User.query.get(custom_id):
        return jsonify({"error": "A user with this ID already exists"}), 400

    # --- Role specific handling ---

    # ====== Student ======
    if role == "student":
        course = data.get("course")
        section = data.get("section")

        if not all([course, section]):
            return jsonify({"error": "Students must include course and section"}), 400

        # Check if student already exists
        if Student.query.get(custom_id):
            return jsonify({"error": "Student with this ID already exists"}), 400

        new_student = Student(
            id=custom_id,   # USE YOUR PROVIDED ID
            name=name,
            course=course,
            section=section
        )
        db.session.add(new_student)

        # Add to section mapping table
        section_map = SectionStudent(
            section=section,
            student_id=custom_id
        )
        db.session.add(section_map)

    # ====== Teacher ======
    elif role == "teacher":
        subject = data.get("subject")

        if not subject:
            return jsonify({"error": "Teachers must include subject"}), 400

        # Check if teacher already exists
        if Teacher.query.get(custom_id):
            return jsonify({"error": "Teacher with this ID already exists"}), 400

        new_teacher = Teacher(
            id=custom_id,   # USE YOUR PROVIDED ID
            name=name,
            subject=subject
        )
        db.session.add(new_teacher)

    # ====== Admin ======
    elif role == "admin":
        pass  # Admin has no student/teacher table

    else:
        return jsonify({"error": "Invalid role"}), 400

    # ====== Create User Account ======
    user = User(
        id=custom_id,      # SAME ID used as USER ID
        role=role
    )
    user.set_password(password)
    db.session.add(user)

    # ====== Save everything ======
    db.session.commit()

    return jsonify({"message": f"{role.capitalize()} added successfully!"}), 200

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

SIMILARITY_THRESHOLD = 0.50

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.all(a == 0) or np.all(b == 0):
        return -1.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))
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

@app.route("/recognize_and_mark", methods=["POST"])
def recognize_and_mark():

    # 1) Validate inputs
    if "image" not in request.files:
        return jsonify({"error": "image file is required (form-data key: image)"}), 400

    try:
        subject_id = int(request.form.get("subject_id"))
    except:
        return jsonify({"error": "subject_id is required and must be numeric"}), 400

    # Check subject exists
    subject = Subject.query.get(subject_id)
    if not subject:
        return jsonify({"error": f"Subject id {subject_id} not found"}), 404

    # Fetch teacher automatically from subject
    teacher_id = subject.teacher_id

    # Read image
    image_file = request.files["image"]
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image or unable to decode"}), 400

    # 2) Extract embedding BEFORE touching DB
    try:
        rep = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            detector_backend="mtcnn"
        )
        probe_embedding = rep[0]["embedding"]
    except Exception as e:
        return jsonify({"error": "Face encoding failed", "details": str(e)}), 400

    # 3) Load all existing encodings
    enc_rows = FacialEncoding.query.all()
    if not enc_rows:
        return jsonify({"error": "No face encodings in database yet"}), 400

    student_encs = {}
    for r in enc_rows:
        sid = r.student_id
        emb = np.array(r.encoding, dtype=np.float32)
        student_encs.setdefault(sid, []).append(emb)

    # 4) Find best match
    best_student = None
    best_score = -1.0

    for sid, embeddings in student_encs.items():
        scores = [cosine_similarity(probe_embedding, e) for e in embeddings]
        max_score = max(scores)

        if max_score > best_score:
            best_score = max_score
            best_student = sid

    # 5) Unknown face
    if best_score < SIMILARITY_THRESHOLD or best_student is None:
        return jsonify({
            "result": "unknown",
            "message": "No matching student found",
            "best_score": best_score
        }), 200

    # 6) SECTION CHECK (IMPORTANT)
    selected_section = request.form.get("section")
    student_obj = Student.query.get(best_student)

    if student_obj.section != selected_section:
        return jsonify({
            "result": "wrong_section",
            "message": f"Student belongs to section {student_obj.section}, but selected section is {selected_section}"
        }), 200

    # 7) Prevent double-marking
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

    # 8) Insert attendance
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

    # 9) Success response
    return jsonify({
        "result": "present",
        "student_id": best_student,
        "student_name": student_obj.name,
        "score": best_score,
        "subject_id": subject_id,
        "date": today_str
    }), 200
@app.route("/teacher/<int:teacher_id>/subject", methods=["GET"])
def get_teacher_subject(teacher_id):
    subject = Subject.query.filter_by(teacher_id=teacher_id).first()

    if not subject:
        return jsonify({"error": "Subject not found for this teacher"}), 404

    return jsonify({
        "id": subject.id,
        "subject_name": subject.subject_name,
        "section": subject.section
    }), 200

@app.route("/subject/<int:subject_id>/students", methods=["GET"])
def get_subject_students(subject_id):
    subject = Subject.query.get(subject_id)
    if not subject:
        return jsonify({"error": "Subject not found"}), 404

    students = Student.query.filter_by(section=subject.section).all()

    result = [
        {"id": s.id, "name": s.name, "section": s.section}
        for s in students
    ]

    return jsonify(result), 200

@app.route("/student/<int:student_id>", methods=["GET"])
def get_student(student_id):
    student = Student.query.get(student_id)
    if not student:
        return jsonify({"error": "Student not found"}), 404

    return jsonify({
        "id": student.id,
        "name": student.name,
        "section": student.section,
        "course": student.course
    }), 200

@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    data = request.get_json()

    # 1) Validate fields are present
    required_fields = ["student_id", "subject_id", "status", "date"]
    for f in required_fields:
        if f not in data:
            return jsonify({"error": f"Missing field '{f}'"}), 400

    try:
        student_id = int(data["student_id"])
        subject_id = int(data["subject_id"])
        status = data["status"]
        date_value = data["date"]     # "2025-11-20"
    except:
        return jsonify({"error": "Invalid or incorrectly formatted fields"}), 400

    # 2) Validate student exists
    student = Student.query.get(student_id)
    if not student:
        return jsonify({"error": f"Student {student_id} not found"}), 404

    # 3) Validate subject exists
    subject = Subject.query.get(subject_id)
    if not subject:
        return jsonify({"error": f"Subject {subject_id} not found"}), 404

    # 4) Automatically fetch teacher_id from Subject table
    teacher_id = subject.teacher_id

    # 5) OPTIONAL: Block marking students from wrong section
    if student.section != subject.section:
        return jsonify({
            "error": "Student not in this subject's section",
            "student_section": student.section,
            "subject_section": subject.section
        }), 400

    # 6) Prevent double marking
    old = Attendance.query.filter_by(
        student_id=student_id,
        subject_id=subject_id,
        date=date_value
    ).first()

    if old:
        return jsonify({
            "result": "already_marked",
            "message": "Attendance already marked for this student today"
        }), 200

    # 7) Insert attendance
    try:
        new_record = Attendance(
            student_id=student_id,
            subject_id=subject_id,
            teacher_id=teacher_id,   # Fetched automatically
            date=date_value,
            status=status
        )
        db.session.add(new_record)
        db.session.commit()

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Database write failed", "details": str(e)}), 500

    # 8) Success response
    return jsonify({
        "result": "success",
        "message": "Attendance marked successfully",
        "student_id": student_id,
        "subject_id": subject_id,
        "teacher_id": teacher_id,
        "status": status,
        "date": date_value
    }), 200


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)


# git add .
# git commit -m ""
# git push

