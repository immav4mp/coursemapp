from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import math, hashlib, json, os, random
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from utils import load_questions, save_questions, load_training_data, save_training_data
from collections import defaultdict
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sqlalchemy import text
from flask_sqlalchemy import SQLAlchemy
from werkzeug.routing import BuildError

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Make this a secure random string in production

# ✅ Database config for Render PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---- Template Filters ----
@app.template_filter('format_datetime')
def format_datetime(value):
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
        return dt.strftime("%b %d, %Y %I:%M %p")
    except:
        return value

@app.template_filter('todatetime')
def to_datetime(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except:
        return value

# ---- Home Page ----
@app.route('/')
def index():
    user = session.get('user')
    is_new_user = session.pop('is_new_user', False)
    return render_template('index.html', user=user, is_new_user=is_new_user, year=datetime.now().year)

# ---- LOGIN ----
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()

        user = db.session.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": email}
        ).mappings().first()

        if user and user['password'] == password:
            session['user'] = {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'full_name': user['full_name'],
                'strand': user['strand'],
                'profile_completed': user['profile_completed'],
                'role': user['role'],
                'is_admin': (user['role'] == 'admin')
            }

            # Load history JSON
            history_file = f"user_history/{email}.json"
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                history = []
            session['user']["history"] = history
            session['history'] = history
            session['is_new_user'] = (request.args.get('from_reg') == '1')

            now = datetime.now()
            existing_log = db.session.execute(text("""
                SELECT * FROM user_logs 
                WHERE user_id = :uid AND logout_time IS NULL
                ORDER BY login_time DESC LIMIT 1
            """), {"uid": user['id']}).mappings().first()

            if existing_log:
                db.session.execute(text("""
                    UPDATE user_logs SET login_time = :t WHERE id = :id
                """), {"t": now, "id": existing_log['id']})
            else:
                db.session.execute(text("""
                    INSERT INTO user_logs (user_id, login_time) VALUES (:uid, :t)
                """), {"uid": user['id'], "t": now})
            db.session.commit()

            return redirect(url_for('admin_dashboard') if user['role'] == 'admin' else url_for('index', alert='login_success'))
        else:
            return redirect(url_for('index', alert='login_failed'))
    return redirect(url_for('index'))

# ---- REGISTER ----
@app.route('/register', methods=['POST'])
def register():
    first_name = request.form['first_name'].strip()
    last_name = request.form['last_name'].strip()
    email = request.form['email'].strip()
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        return redirect(url_for('index', register_alert='password_mismatch'))

    full_name = f"{first_name} {last_name}"
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    account = db.session.execute(
        text("SELECT * FROM users WHERE email = :email"),
        {"email": email}
    ).mappings().first()

    if account:
        return redirect(url_for('index', register_alert='email_exists'))
    else:
        db.session.execute(text("""
            INSERT INTO users (first_name, last_name, full_name, email, password, profile_completed)
            VALUES (:fn, :ln, :full, :email, :pw, FALSE)
        """), {"fn": first_name, "ln": last_name, "full": full_name, "email": email, "pw": hashed_password})
        db.session.commit()
        return redirect(url_for('index', register_alert='success'))


#profile test
@app.route('/complete_profile', methods=['GET', 'POST'])
def complete_profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    if session['user'].get('profile_completed') == 1:
        return redirect(url_for('index'))

    user_id = session['user']['id']

    if request.method == 'POST':
        strand = request.form['strand']
        db.session.execute(text("""
            UPDATE users SET strand = :s, profile_completed = TRUE WHERE id = :uid
        """), {"s": strand, "uid": user_id})
        db.session.commit()

        updated_user = db.session.execute(
            text("SELECT * FROM users WHERE id = :uid"), {"uid": user_id}
        ).mappings().first()

        session['user'].update({
            'strand': updated_user['strand'],
            'profile_completed': updated_user['profile_completed']
        })
        session['is_new_user'] = True
        flash('Profile completed successfully.', 'success')
        return redirect(url_for('index'))

    return render_template('complete_profile.html', user=session['user'])

# Dashboard Page
# ---- Dashboard ----
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    if not user.get('profile_completed'):
        return redirect(url_for('complete_profile'))
    return render_template('dashboard.html', user=user)

# --- KNOWLEDGE TEST ---
@app.route('/course_test/knowledge', methods=['GET', 'POST'])
def course_test_knowledge():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()
    question_bank = load_questions()
    strand_data = question_bank.get(strand, {})
    knowledge_questions = strand_data.get('knowledge', [])

    if request.method == 'POST':
        submitted = request.form.to_dict()  # { "q101": "A", "q102": "B", ... }
        answers = {}
        correct_count = 0

        for q in knowledge_questions:
            qid = str(q["id"])
            user_answer = submitted.get(f"q{qid}")
            if user_answer:
                answers[qid] = user_answer.upper()
                if user_answer.upper() == q.get("answer"):
                    correct_count += 1

        # Save to session
        session['knowledge_answers'] = answers
        session['knowledge_score'] = correct_count
        session['knowledge_total'] = len(knowledge_questions)
        return redirect(url_for('course_test_interest'))

    # Pick 30 random questions
    random.shuffle(knowledge_questions)
    knowledge_questions = knowledge_questions[:30]

    return render_template(
        'course_test_knowledge.html',
        knowledge_questions=knowledge_questions,
        user=user
    )


# --- INTEREST TEST ---
@app.route('/course_test/interest', methods=['GET', 'POST'])
def course_test_interest():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()

    # Load interest bank
    with open('interest_bank.json', 'r') as f:
        interest_bank = json.load(f)

    strand_data = interest_bank.get(strand, {}).get("interest", [])

    # Group questions by course
    course_map = {}
    for q in strand_data:
        course_map.setdefault(q["course"], []).append(q)

    # Select exactly 1 random question per course
    strand_questions = [random.choice(q_list) for q_list in course_map.values()]

    if request.method == 'POST':
        answers = {}
        for q in strand_questions:
            ans = request.form.get(f"q_{q['id']}")
            if ans:
                answers[q["course"]] = int(ans)  # save as numeric 1–5

        session['interest_scores'] = answers
        return redirect(url_for('course_test_aptitude'))

    return render_template(
        'course_test_interest.html',
        strand=strand.capitalize(),
        questions=strand_questions
    )


# --- APTITUDE TEST ---
@app.route('/course_test/aptitude', methods=['GET', 'POST'])
def course_test_aptitude():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()

    # Load aptitude questions from separate file
    aptitude_bank = load_questions("aptitude_bank.json")
    strand_data = aptitude_bank.get(strand, {})
    all_questions = strand_data.get("aptitude", [])

    # Pick only 5 random questions
    import random
    selected_questions = random.sample(all_questions, min(5, len(all_questions)))

    if request.method == 'POST':
        submitted = request.form.to_dict()
        correct_count = 0

        # Check only from selected questions
        for q in selected_questions:
            qid = str(q["id"])
            user_answer = submitted.get(f"q{qid}")
            if user_answer and user_answer.upper() == q.get("answer"):
                correct_count += 1

        # Save score in session
        session['aptitude_score'] = correct_count
        session['aptitude_total'] = len(selected_questions)
        return redirect(url_for('course_test_personality'))

    return render_template(
        'course_test_aptitude.html',
        aptitude_questions=selected_questions,
        user=user
    )


# --- PERSONALITY TEST ---
@app.route('/course_test/personality', methods=['GET', 'POST'])
def course_test_personality():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()

    # Load personality questions from JSON
    with open("personality_bank.json", "r") as f:
        personality_bank = json.load(f)

    # Get strand-specific questions (default empty list if not found)
    all_questions = personality_bank.get(strand, {}).get("personality", [])

    # Randomly pick 5 unique questions
    questions = random.sample(all_questions, min(5, len(all_questions)))

    if request.method == 'POST':
        session['personality_answers'] = request.form.to_dict()
        return redirect(url_for('course_test_goal'))

    return render_template("course_test_personality.html", questions=questions, user=user)


# ---GOAL TEST ---
@app.route('/course_test/goal', methods=['GET', 'POST'])
def course_test_goal():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()

    # Load goal_bank.json
    with open("goal_bank.json", "r") as f:
        goal_bank = json.load(f)

    # Pick 5 random goals from the user's strand
    all_goals = goal_bank.get(strand, [])
    selected_goals = random.sample(all_goals, min(5, len(all_goals)))

    if request.method == 'POST':
        session['goal_answers'] = request.form.getlist('goals')
        return redirect(url_for('show_result'))

    return render_template("course_test_goal.html", goals=selected_goals, user=user)



# test submit
from collections import defaultdict
@app.route('/submit_test', methods=['POST'])
def submit_test():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()
    questions = load_questions()
    strand_data = questions.get(strand, {})

    #knowledge
    knowledge_questions = strand_data.get("knowledge", [])
    total_knowledge = len(knowledge_questions)
    correct_knowledge = 0
    course_correct_counts = defaultdict(int)
    course_total_counts = defaultdict(int)

    for q in knowledge_questions:
        qid = str(q["id"])
        user_answer = request.form.get(f"q{qid}")
        correct_answer = q.get("answer")
        course = q.get("course")

        if course:
            course_total_counts[course] += 1
            if user_answer and user_answer.upper() == correct_answer:
                correct_knowledge += 1
                course_correct_counts[course] += 1

    raw_knowledge_score = (correct_knowledge / total_knowledge) * 100 if total_knowledge > 0 else 0
    knowledge_weight = 0.35
    knowledge_score = raw_knowledge_score * knowledge_weight

    #others scores
    def get_weighted_score(key, weight):
        raw = float(request.form.get(key, 100))
        return raw, (raw / 100) * weight

    interest_raw, interest_score = get_weighted_score("interest_score", 20)
    aptitude_raw, aptitude_score = get_weighted_score("aptitude_score", 20)
    personality_raw, personality_score = get_weighted_score("personality_score", 15)
    goal_raw, goal_score = get_weighted_score("goal_score", 10)

    final_score = round(
        knowledge_score + interest_score + aptitude_score + personality_score + goal_score,
        2
    )

    #naive bayes sample scoring
    course_probabilities = {}

    for course in course_total_counts:
        #simulated P(course) = uniform for simplicity
        prior = 1  

        #simulated likelihoods based on score
        knowledge_likelihood = (course_correct_counts[course] / course_total_counts[course]) if course_total_counts[course] > 0 else 0.01
        interest_likelihood = (interest_raw / 100) or 0.01
        aptitude_likelihood = (aptitude_raw / 100) or 0.01
        personality_likelihood = (personality_raw / 100) or 0.01
        goal_likelihood = (goal_raw / 100) or 0.01

        #multiply likelihoods
        log_prob = (
            math.log(prior) +
            knowledge_weight * math.log(knowledge_likelihood) +
            0.20 * math.log(interest_likelihood) +
            0.20 * math.log(aptitude_likelihood) +
            0.15 * math.log(personality_likelihood) +
            0.10 * math.log(goal_likelihood)
        )

        course_probabilities[course] = log_prob

    suggested_course = max(course_probabilities, key=course_probabilities.get) if course_probabilities else None

    return render_template("result.html",
                           score=final_score,
                           suggested_course=suggested_course,
                           user=user)

@app.route('/submit_knowledge', methods=['POST'])
def submit_knowledge():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()
    questions = load_questions()
    knowledge_questions = questions.get(strand, {}).get("knowledge", [])

    course_correct = {}
    course_total = {}

    for q in knowledge_questions:
        qid = str(q["id"])
        selected = request.form.get(f"q{qid}")
        course = q.get("course")

        if course:
            course_total[course] = course_total.get(course, 0) + 1
            if selected and selected.upper() == q.get("answer"):
                course_correct[course] = course_correct.get(course, 0) + 1

    #Store percentage per course
    course_scores = {
        course: round((course_correct.get(course, 0) / total) * 100, 2)
        for course, total in course_total.items()
    }

    session['knowledge_score'] = round(
        sum(course_scores.values()) / len(course_scores), 2
    ) if course_scores else 0

    session['course_knowledge_scores'] = course_scores
    return redirect(url_for('course_test_interest'))

# --- RESULT CALCULATION: NAIVE BAYES + ADVISER FORMULA ---
from flask import render_template, session, redirect, url_for
from sklearn.naive_bayes import GaussianNB
from utils import load_training_data, load_questions
from datetime import datetime
import os, json, random
import numpy as np

@app.route('/show_result')
def show_result():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    strand = user['strand'].lower()
    email = user['email']

    # --- get stored answers ---
    knowledge_answers = session.get('knowledge_answers', {})  # {question_id: chosen_option}
    knowledge_total = session.get('knowledge_total', 30)

    aptitude_score = session.get('aptitude_score', 0)
    aptitude_total = session.get('aptitude_total', 5)

    interest_answers = session.get('interest_answers', {})
    personality_answers = session.get('personality_answers', {})
    goal_answers = session.get('goal_answers', [])

    # --- load data ---
    all_training_data = load_training_data()
    strand_training_data = all_training_data.get(strand, [])

    question_bank = load_questions()
    strand_questions = question_bank.get(strand, {})

    # ====================================================
    # 1) KNOWLEDGE: Course-based ranking
    # ====================================================
    knowledge_correct_per_course = {}
    knowledge_score = 0

    for q in strand_questions.get("knowledge", []):
        qid = str(q["id"])
        if qid in knowledge_answers and knowledge_answers[qid] == q["answer"]:
            knowledge_score += 1
            course = q["course"]
            knowledge_correct_per_course[course] = knowledge_correct_per_course.get(course, 0) + 1

    # Rank courses by most correct
    ranked_knowledge_courses = sorted(
        knowledge_correct_per_course.items(),
        key=lambda x: x[1],
        reverse=True
    )

    knowledge_top3 = ranked_knowledge_courses[:3]
    suggested_course = knowledge_top3[0][0] if knowledge_top3 else "No course matched"

    # --- percent scores ---
    knowledge_percent = round((knowledge_score / knowledge_total) * 100, 1) if knowledge_total > 0 else 0
    aptitude_percent = round((aptitude_score / aptitude_total) * 100, 1) if aptitude_total > 0 else 0
    interest_score = len([v for v in interest_answers.values() if v]) * 10
    personality_score = len([v for v in personality_answers.values() if v]) * 20
    goal_score = len(goal_answers) * 20

    course_data = []
    adviser_course_data = []

    # ====================================================
    # 2) NAIVE BAYES + ADVISER for Aptitude, Interest, Personality, Goal
    # ====================================================
    if strand_training_data:
        X_train = np.array([item['features'] for item in strand_training_data], dtype=float)
        y_train = np.array([item['course'] for item in strand_training_data])

        model = GaussianNB()
        model.fit(X_train, y_train)

        user_features = np.array([[knowledge_percent, interest_score, aptitude_percent, personality_score, goal_score]], dtype=float)

        probas = model.predict_proba(user_features)[0]
        classes = model.classes_

        ranked = sorted(zip(classes, probas), key=lambda x: x[1], reverse=True)
        top_ranked = ranked[:3]

        # normalize + jitter
        raw = [p for _, p in top_ranked]
        total_raw = sum(raw) or 1
        normalized = [p / total_raw for p in raw]

        jittered = []
        for val in normalized:
            variation = random.uniform(-0.1, 0.1)
            jittered.append(max(0.05, val + variation))

        total_jitter = sum(jittered)
        final = [(val / total_jitter) * 100 for val in jittered]

        for (course, _), pct in zip(top_ranked, final):
            percentage = round(pct, 1)
            explanation = f"This course matches your profile with {percentage}% confidence (Naïve Bayes)."
            course_data.append((course, explanation, percentage))

        # Adviser formula
        adviser_scores = []
        for course in classes:
            I_c = 100 if interest_answers.get(course) else 0
            P_c = personality_score
            adv_score = (
                0.35 * knowledge_percent +
                0.20 * I_c +
                0.20 * aptitude_percent +
                0.15 * P_c +
                0.10 * goal_score
            )
            adviser_scores.append((course, adv_score))

        total_adv = sum(s for _, s in adviser_scores) or 1
        adviser_scores = [(c, (s / total_adv) * 100) for c, s in adviser_scores]

        adviser_top3 = sorted(adviser_scores, key=lambda x: x[1], reverse=True)[:3]
        for course, pct in adviser_top3:
            percentage = round(pct, 1)
            explanation = f"This course matches your profile with {percentage}% based on adviser formula."
            adviser_course_data.append((course, explanation, percentage))

    # ====================================================
    # 3) Save Result History
    # ====================================================
    history_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "suggested_course": suggested_course,
        "knowledge_top3": [c for c, _ in knowledge_top3],
        "scores": {
            "knowledge": f"{knowledge_score}/{knowledge_total}",
            "interest": f"{interest_score}%",
            "aptitude": f"{aptitude_score}/{aptitude_total}",
            "personality": f"{personality_score}%",
            "goal": f"{goal_score}%"
        }
    }

    os.makedirs("user_history", exist_ok=True)
    history_file = f"user_history/{email}.json"
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

    history.append(history_entry)

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # update session
    user["history"] = history
    session["user"] = user
    session["history"] = history

    return render_template(
        "result.html",
        user=user,
        knowledge_score=f"{knowledge_score}/{knowledge_total}",
        interest_score=f"{interest_score}%",
        aptitude_score=f"{aptitude_score}/{aptitude_total}",
        personality_score=f"{personality_score}%",
        goal_score=f"{goal_score}%",
        suggested_course=suggested_course,
        knowledge_top3=knowledge_top3,
        course_data=course_data,
        adviser_course_data=adviser_course_data
    )

#saved

import os, json
from flask import render_template, session, redirect, url_for
from datetime import datetime

# Add Jinja2 filter to convert string to datetime
@app.template_filter('todatetime')
def to_datetime(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except:
        return value

#history route
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    email = user['email']

    # ✅ Always reload history from file
    history_file = f"user_history/{email}.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []

    # Update session too (so it's synced)
    session['user']["history"] = history
    session['history'] = history

    return render_template("history.html", user=user, history=history)


# Delete history record
@app.route('/delete_history/<int:index>', methods=['POST'])
def delete_history(index):
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    email = user['email']

    # Load history from file
    history_file = f"user_history/{email}.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []

    # Remove the record if index is valid
    if 0 <= index < len(history):
        history.pop(index)

        # Save back to file
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        # Update session copy too
        session['user']['history'] = history
        session['history'] = history

    return redirect(url_for('history'))


# ---- Admin Dashboard Example (converted) ----
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user' not in session or session['user'].get('role') != 'admin':
        return redirect(url_for('login'))

    # Load JSON banks
    try:
        with open('question_bank.json', 'r') as f: question_bank = json.load(f)
    except: question_bank = {}
    try:
        with open('training_data.json', 'r') as f: training_data = json.load(f)
    except: training_data = {}

    total_users = db.session.execute(text("SELECT COUNT(*) FROM users")).scalar()
    total_admins = db.session.execute(text("SELECT COUNT(*) FROM users WHERE role = 'admin'")).scalar()
    total_students = db.session.execute(text("SELECT COUNT(*) FROM users WHERE role = 'student'")).scalar()

    strand_counts = db.session.execute(text("""
        SELECT strand, COUNT(*) AS count FROM users
        WHERE role = 'student' GROUP BY strand
    """)).mappings().all()

    strand_distribution = {}
    if total_students > 0:
        for row in strand_counts:
            strand = row['strand'] or "Unspecified"
            strand_distribution[strand] = round((row['count'] / total_students) * 100, 2)

    users = db.session.execute(
        text("SELECT * FROM users WHERE role = 'student")
    ).mappings().all()

    return render_template("admin_dashboard.html",
        users=users,
        question_bank=question_bank,
        training_data=training_data,
        total_users=total_users,
        total_admins=total_admins,
        total_students=total_students,
        strand_distribution=strand_distribution,
        current_user_id=session['user']['id']
    )





# --- ADD QUESTION ---
@app.route('/admin/question/add', methods=['GET', 'POST'])
def add_question():
    if 'user' not in session or session['user'].get('role') != 'admin':
        return redirect(url_for('login'))

    try:
        data = load_questions()
    except Exception:
        data = {}

    if request.method == 'POST':
        strand = request.form['strand'].lower()
        category = request.form['category'].lower()

        # ✅ Ensure structure
        if strand not in data:
            data[strand] = {c: [] for c in ["knowledge", "interest", "aptitude", "personality", "goal"]}
        elif category not in data[strand]:
            data[strand][category] = []

        # ✅ Auto-generate next ID (store as string)
        existing_ids = [int(q["id"]) for q in data[strand][category] if str(q.get("id", "")).isdigit()]
        next_id = str(max(existing_ids, default=0) + 1)

        # ✅ New question object
        new_question = {
            "id": next_id,
            "course": request.form['course'],
            "question": request.form['question'],
            "option_a": request.form['option_a'],
            "option_b": request.form['option_b'],
            "option_c": request.form['option_c'],
            "option_d": request.form['option_d'],
            "answer": request.form['answer'].upper()
        }

        # ✅ Save
        data[strand][category].append(new_question)
        save_questions(data)

        flash("✅ Question added successfully!", "success")
        return redirect(url_for('admin_dashboard'))

    # ✅ For GET → show next global available ID
    all_ids = [int(q["id"]) for strand in data.values() for cat in strand.values() for q in cat if str(q.get("id", "")).isdigit()]
    next_id = str(max(all_ids, default=0) + 1)

    return render_template('add_question.html', question_bank=data, next_id=next_id)



# --- EDIT QUESTION ---
@app.route('/admin/question/edit/<qid>', methods=['GET', 'POST'])
def edit_question(qid):
    if 'user' not in session or session['user'].get('role') != 'admin':
        return redirect(url_for('login'))

    data = load_questions()
    question_to_edit = None
    strand_key, category_key = None, None

    # ✅ Search through all strands and categories
    for strand, categories in data.items():
        for category, questions in categories.items():
            for q in questions:
                if str(q.get("id")) == str(qid):   # compare as string
                    question_to_edit = q
                    strand_key, category_key = strand, category
                    break
            if question_to_edit:
                break
        if question_to_edit:
            break

    if not question_to_edit:
        flash("⚠️ Question not found!", "danger")
        return redirect(url_for('admin_dashboard'))

    # ✅ Ensure fields exist
    question_to_edit.setdefault("course", "")
    question_to_edit.setdefault("option_a", "")
    question_to_edit.setdefault("option_b", "")
    question_to_edit.setdefault("option_c", "")
    question_to_edit.setdefault("option_d", "")

    if request.method == 'POST':
        # ✅ Update fields from form
        question_to_edit.update({
            "question": request.form['question'],
            "course": request.form['course'],
            "option_a": request.form['option_a'],
            "option_b": request.form['option_b'],
            "option_c": request.form['option_c'],
            "option_d": request.form['option_d'],
            "answer": request.form['answer'].upper()
        })

        save_questions(data)
        flash("✅ Question updated successfully!", "success")
        return redirect(url_for('admin_dashboard'))

    return render_template(
        'edit_question.html',
        question=question_to_edit,
        strand=strand_key,
        category=category_key
    )


# --- DELETE QUESTION ---
@app.route('/admin/question/delete/<qid>')   # ✅ treat ID as string
def delete_question(qid):
    if 'user' not in session or session['user'].get('role') != 'admin':
        return redirect(url_for('login'))

    data = load_questions()
    found = False

    for strand in data:
        for category in data[strand]:
            original_len = len(data[strand][category])
            # ✅ Compare as string, since JSON IDs are strings
            data[strand][category] = [q for q in data[strand][category] if str(q["id"]) != str(qid)]
            if len(data[strand][category]) < original_len:
                found = True
                break
        if found:
            break

    if found:
        save_questions(data)
        flash("🗑️ Question deleted successfully!", "success")
    else:
        flash("⚠️ Question not found!", "danger")

    return redirect(url_for('admin_dashboard'))

# --- BULK DELETE (Generic Handler) ---
@app.route('/admin/delete_selected/<entity>', methods=['POST'])
def delete_selected(entity):
    if 'user' not in session or session['user'].get('role') != 'admin':
        return redirect(url_for('login'))

    selected = request.form.getlist('selected')
    if not selected:
        flash("⚠️ No items selected.", "error")
        return redirect(url_for('admin_dashboard'))

    deleted = 0

    # --- Questions (JSON) ---
    if entity == "questions":
        data = load_questions()
        strand = request.form.get('strand')
        category = request.form.get('category')
        if strand and category:
            before = len(data[strand][category])
            data[strand][category] = [
                q for q in data[strand][category] if str(q["id"]) not in selected
            ]
            deleted = before - len(data[strand][category])
            save_questions(data)

    # --- Users (MySQL DB) ---
    elif entity == "users":
        cursor = mysql.connection.cursor()
        format_strings = ','.join(['%s'] * len(selected))
        query = f"DELETE FROM users WHERE id IN ({format_strings})"
        cursor.execute(query, tuple(selected))
        mysql.connection.commit()
        deleted = cursor.rowcount
        cursor.close()

    # --- Training Data (JSON) ---
    elif entity == "training":
        strand = request.form.get('strand')
        data = load_training_data()
        indexes = set(map(int, selected))
        before = len(data[strand])
        data[strand] = [entry for i, entry in enumerate(data[strand]) if i not in indexes]
        deleted = before - len(data[strand])
        save_training_data(data)

    else:
        flash("⚠️ Invalid entity type.", "danger")
        return redirect(url_for('admin_dashboard'))

    if deleted > 0:
        flash(f"🗑️ Deleted {deleted} {entity} item(s) successfully!", "success")
    else:
        flash("⚠️ No matching items found.", "danger")

    return redirect(url_for('admin_dashboard'))



import hashlib

# --- Edit Users ---
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if request.method == 'POST':
        new_password = request.form['new_password']

        # ✅ Hash the new password before saving
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()

        cursor.execute("""
            UPDATE users
            SET password = %s
            WHERE id = %s
        """, (hashed_password, user_id))
        mysql.connection.commit()
        cursor.close()

        flash('Password updated successfully.', 'success')
        return redirect(url_for('admin_dashboard'))

    else:
        # Fetch user info for display
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if not user:
            flash("User not found.", "error")
            return redirect(url_for('admin_dashboard'))

        return render_template('edit_user.html', user=user)



@app.route('/delete_users', methods=['POST'])
def delete_users():
    selected_users = request.form.getlist('selected_users')  # list of IDs from checkboxes
    if not selected_users:
        flash('No users selected.', 'warning')
        return redirect(url_for('admin_dashboard'))

    cursor = mysql.connection.cursor()
    # Build placeholder string for SQL IN clause
    placeholders = ','.join(['%s'] * len(selected_users))
    cursor.execute(f"DELETE FROM users WHERE id IN ({placeholders})", tuple(selected_users))
    mysql.connection.commit()
    cursor.close()

    flash(f'{len(selected_users)} user(s) deleted successfully.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/delete_user/<int:user_id>', methods=['GET'])
def delete_user(user_id):
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    mysql.connection.commit()
    cursor.close()
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin_dashboard'))



# --- ADD TRAINING DATA ---
@app.route('/admin/training_data/add', methods=['GET', 'POST'])
def add_training_data():
    if request.method == 'POST':
        strand = request.form['strand'].lower()
        course = request.form['course']
        features = request.form.getlist('features')

        training_data = load_training_data()
        if strand not in training_data:
            training_data[strand] = []

        training_data[strand].append({
            'course': course,
            'features': features
        })

        save_training_data(training_data)
        flash("✅ Training data added successfully!", "success")
        return redirect(url_for('admin_dashboard'))

    return render_template('add_training_data.html')


# --- EDIT TRAINING DATA ---
@app.route('/admin/training_data/edit/<strand>/<int:index>', methods=['GET', 'POST'])
def edit_training_data(strand, index):
    data = load_training_data()
    strand = strand.lower()

    # Protect against out-of-range
    if strand not in data or index >= len(data[strand]):
        flash("⚠️ Training data not found.", "error")
        return redirect(url_for('admin_dashboard'))

    entry = data[strand][index]

    if request.method == 'POST':
        entry['course'] = request.form['course']
        entry['features'] = [f.strip() for f in request.form['features'].split(',')]

        save_training_data(data)
        flash("✏️ Training data updated successfully!", "success")
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_training_data.html', strand=strand, index=index, entry=entry)


# --- DELETE TRAINING DATA (Single) ---
@app.route('/admin/training_data/delete/<strand>/<int:index>')
def delete_training_data(strand, index):
    data = load_training_data()
    strand = strand.lower()

    if strand in data and index < len(data[strand]):
        del data[strand][index]
        save_training_data(data)
        flash("🗑️ Training data deleted successfully!", "success")
    else:
        flash("⚠️ Training data not found.", "error")

    return redirect(url_for('admin_dashboard'))


# --- BULK DELETE TRAINING DATA ---
@app.route('/admin/training_data/delete_selected/<strand>', methods=['POST'])
def delete_selected_training_data(strand):
    if 'user' not in session or session['user'].get('role') != 'admin':
        return redirect(url_for('login'))

    indexes = request.form.getlist('indexes')
    if not indexes:
        flash("⚠️ No training data selected.", "error")
        return redirect(url_for('admin_dashboard'))

    data = load_training_data()
    strand = strand.lower()

    if strand not in data:
        flash("⚠️ Strand not found.", "error")
        return redirect(url_for('admin_dashboard'))

    indexes = set(map(int, indexes))  # convert to ints
    before = len(data[strand])
    data[strand] = [entry for i, entry in enumerate(data[strand]) if i not in indexes]
    deleted = before - len(data[strand])

    if deleted > 0:
        save_training_data(data)
        flash(f"🗑️ Deleted {deleted} training data entry(ies) successfully!", "success")
    else:
        flash("⚠️ No matching entries found.", "danger")

    return redirect(url_for('admin_dashboard'))




#edit profile
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
import os, MySQLdb

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']

    if request.method == 'POST':
        # Get form data
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Keep existing values if field is blank
        if not full_name:
            full_name = user.get('full_name', '')
        if not email:
            email = user.get('email', '')

        # Handle password (hash only if provided, else keep existing)
        if password:
            hashed_password = generate_password_hash(password)
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("SELECT password FROM users WHERE id = %s", (user['id'],))
            db_user = cursor.fetchone()
            hashed_password = db_user['password'] if db_user else ''
            cursor.close()

        # Update user in database (only full_name, email, password)
        cursor = mysql.connection.cursor()
        cursor.execute("""
            UPDATE users 
            SET full_name = %s, email = %s, password = %s
            WHERE id = %s
        """, (full_name, email, hashed_password, user['id']))
        mysql.connection.commit()
        cursor.close()

        # Update session data
        user.update({
            'full_name': full_name,
            'email': email,
        })
        session['user'] = user

        return redirect(url_for('index'))

    # Build full_name if empty from first_name + last_name
    if not user.get('full_name'):
        user['full_name'] = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()

    return render_template('edit_profile.html', user=user)



# --- LOGOUT ROUTE ---
@app.route('/logout')
def logout():
    if 'user' in session:
        user_id = session['user']['id']
        db.session.execute(text("""
            UPDATE user_logs
            SET logout_time = :t
            WHERE id = (
                SELECT id FROM user_logs 
                WHERE user_id = :uid AND logout_time IS NULL
                ORDER BY login_time DESC LIMIT 1
            )
        """), {"t": datetime.now(), "uid": user_id})
        db.session.commit()
        session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


#Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
