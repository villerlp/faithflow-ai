import os
import json
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from openai import OpenAI
import openai
import stripe

# ============================================
# Flask setup
# ============================================
app = Flask(__name__)
app.secret_key = os.environ.get("FAITHFLOW_SECRET_KEY", "dev-secret-change-me")

# ============================================
# Stripe configuration
# ============================================
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# ============================================
# OpenAI client
# ============================================
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ============================================
# Database setup (SQLite local, DATABASE_URL on Render)
# ============================================
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Render / production DB (e.g. Postgres)
    engine = create_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True)
else:
    # Local SQLite fallback
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DB_PATH = os.path.join(BASE_DIR, "faithflow.db")
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_db():
    """Context manager that always closes the DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================
# Models
# ============================================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Plan / usage
    plan = Column(String(50), default="free")  # "free" or "creator"
    # We no longer rely on monthly_generations, but leave it here if it exists in DB
    monthly_generations = Column(Integer, default=0)

    generations = relationship("Generation", back_populates="user")


class Generation(Base):
    __tablename__ = "generations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    gen_type = Column(String(50), nullable=False)  # "kids_devotional" or "tiktok_script"
    theme = Column(String(255), nullable=False)
    input_data = Column(Text, nullable=False)
    output_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="generations")


Base.metadata.create_all(engine)

# ============================================
# Usage limits
# ============================================
FREE_PLAN_LIMIT = 5  # free users get 5 generations per calendar month


def get_monthly_usage(db, user_id: int) -> int:
    """
    Count how many generations this user has in the **current month**.
    """
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    count = (
        db.query(Generation)
        .filter(Generation.user_id == user_id)
        .filter(Generation.created_at >= month_start)
        .count()
    )
    return count


def can_generate(db, user: User) -> bool:
    """
    True if the user is allowed to generate more content.
    """
    if user.plan == "creator":
        return True
    used = get_monthly_usage(db, user.id)
    return used < FREE_PLAN_LIMIT


# ============================================
# Helpers
# ============================================
def current_user():
    """
    Return the logged-in user object (detached from session),
    or None if not logged in.
    """
    user_id = session.get("user_id")
    if not user_id:
        return None

    with get_db() as db:
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            session.clear()
            return None
        # Detach to avoid issues after session closes
        db.expunge(user)
        return user


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


# ============================================
# Stub generators (fallbacks)
# ============================================
def generate_kids_devotional_stub(theme, age_range, num_days, tone, translation):
    num_map = {"3 days": 3, "5 days": 5, "7 days": 7}
    days = num_map.get(num_days, 3)
    devotional_plan = []
    for i in range(1, days + 1):
        devotional_plan.append(
            {
                "day": i,
                "title": f"{theme} - Day {i}",
                "verse": f"John 14:27 ({translation or 'NIV'})",
                "verse_text": "Peace I leave with you; my peace I give you...",
                "devotional": (
                    f"(Placeholder) This is a sample devotional for {theme}, Day {i}, "
                    f"in a {tone.lower()} tone for {age_range}."
                ),
                "question": "What is one way you can remember that God is with you today?",
                "prayer": "Dear Jesus, thank You for being with me and giving me peace. Amen.",
                "activity": "Draw a picture of a time you felt God’s love or peace.",
            }
        )
    return devotional_plan


def generate_tiktok_script_stub(theme, audience, length, tone, translation):
    script_lines = [
        f"Have you ever felt {theme.lower()}?",
        "You pray, but it feels like nothing is changing.",
        "I want you to know: silence does not mean God is absent.",
    ]

    verses = [
        {
            "reference": f"Psalm 34:18 ({translation or 'NIV'})",
            "text": "The LORD is close to the brokenhearted and saves those who are crushed in spirit.",
            "note": "This reminds us that God stays close in our pain, even when we don’t see it.",
        }
    ]

    caption = (
        f"When it feels like your prayers hit the ceiling, remember: "
        f"God is still listening. #{audience.replace(' ', '')} #faith #hope #Jesus"
    )

    return {
        "hook": "When your prayers feel unanswered, this is for you.",
        "script_lines": script_lines,
        "verses": verses,
        "cta": "If this spoke to you, save this and share it with someone who needs hope today.",
        "caption": caption,
        "length": length,
        "tone": tone,
    }


# ============================================
# AI generators
# ============================================
def generate_kids_devotional_ai(theme, age_range, num_days, tone, translation):
    num_map = {"3 days": 3, "5 days": 5, "7 days": 7}
    days_count = num_map.get(num_days, 3)
    translation = translation or "NIV"

    system_message = (
        "You are a Christian children's author who writes gentle, "
        "scripturally-sound devotions for kids ages 4–12. "
        "All content must be biblically respectful and age-appropriate. "
        "You will return ONLY valid JSON that matches the requested schema. "
        "Do not include any extra commentary, explanation, or text outside of JSON."
    )

    user_prompt = f"""
Create a {days_count}-day devotional plan for children on the theme "{theme}".

Age range: {age_range}
Tone: {tone}
Preferred Bible translation for quoted verses: {translation}

For EACH day, include:
- "day": day number (integer)
- "title": short title for the day (string)
- "verse": Bible reference with translation, e.g. "John 14:27 ({translation})"
- "verse_text": the actual verse text from that translation (string, keep it fairly short)
- "devotional": 120–200 word devotional for kids in this age range, in the given tone
- "question": one simple reflection question for the child
- "prayer": a short 2–4 sentence prayer the child can pray
- "activity": one simple activity idea (drawing, writing, small act of kindness, etc.)

Return a single JSON object with this exact structure:

{{
  "days": [
    {{
      "day": 1,
      "title": "...",
      "verse": "Book chapter:verse ({translation})",
      "verse_text": "...",
      "devotional": "...",
      "question": "...",
      "prayer": "...",
      "activity": "..."
    }}
  ]
}}

IMPORTANT:
- Respond with JSON ONLY.
- Do not include any keys other than "days" at the top level.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    raw_content = completion.choices[0].message.content or ""
    content = raw_content.strip()

    # Strip ```json fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        data = json.loads(content)
        days = data.get("days", [])
    except json.JSONDecodeError:
        days = [
            {
                "day": 1,
                "title": f"{theme} – Devotional Plan",
                "verse": "",
                "verse_text": "",
                "devotional": raw_content,
                "question": "",
                "prayer": "",
                "activity": "",
            }
        ]

    normalized = []
    for d in days:
        normalized.append(
            {
                "day": d.get("day"),
                "title": d.get("title", ""),
                "verse": d.get("verse", ""),
                "verse_text": d.get("verse_text", ""),
                "devotional": d.get("devotional", ""),
                "question": d.get("question", ""),
                "prayer": d.get("prayer", ""),
                "activity": d.get("activity", ""),
            }
        )

    return normalized


def generate_tiktok_script_ai(theme, audience, length, tone, translation):
    translation = translation or "NIV"

    system_message = (
        "You are a Christian content creator and Bible teacher who writes short, "
        "emotionally engaging scripts for TikTok and Reels. "
        "All content must be biblically respectful and Christ-centered. "
        "You will return ONLY valid JSON that matches the requested schema. "
        "Do not include any extra commentary, explanation, or text outside of JSON."
    )

    user_prompt = f"""
Create a short-form video script for TikTok on the theme "{theme}".

Audience: {audience}
Desired length: {length}
Tone: {tone}
Preferred Bible translation for quoted verses: {translation}

The script should:
- Start with a strong hook.
- Use short, natural lines that can be spoken on camera.
- Include 1–2 Bible verses in the requested translation plus a simple explanation.
- End with a hopeful, Christ-centered takeaway.
- Include a simple call to action.
- Include a caption with 5–10 relevant hashtags.

Return this exact JSON structure:

{{
  "hook": "Short opening hook line.",
  "script_lines": [
    "First spoken line.",
    "Second spoken line."
  ],
  "verses": [
    {{
      "reference": "Psalm 34:18 ({translation})",
      "text": "Verse text here...",
      "note": "1–2 sentence explanation of the verse in context of the theme."
    }}
  ],
  "cta": "Short call to action.",
  "caption": "Caption text with hashtags."
}}

IMPORTANT:
- Respond with JSON ONLY.
- Do not include extra keys.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    raw_content = completion.choices[0].message.content or ""
    content = raw_content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "hook": "When your prayers feel unanswered, this is for you.",
            "script_lines": raw_content.splitlines(),
            "verses": [],
            "cta": "",
            "caption": raw_content,
        }

    verses = []
    for v in data.get("verses", []):
        verses.append(
            {
                "reference": v.get("reference", ""),
                "text": v.get("text", ""),
                "note": v.get("note", ""),
            }
        )

    return {
        "hook": data.get("hook", ""),
        "script_lines": data.get("script_lines", []),
        "verses": verses,
        "cta": data.get("cta", ""),
        "caption": data.get("caption", ""),
        "length": length,
        "tone": tone,
    }


# ============================================
# Auth routes
# ============================================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("signup"))

        with get_db() as db:
            existing = db.query(User).filter_by(email=email).first()
            if existing:
                flash("An account with that email already exists.", "danger")
                return redirect(url_for("signup"))

            user = User(
                name=name,
                email=email,
                password_hash=generate_password_hash(password),
            )
            db.add(user)
            db.commit()

        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("auth_signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return redirect(url_for("login"))

        with get_db() as db:
            user = db.query(User).filter_by(email=email).first()
            if not user or not check_password_hash(user.password_hash, password):
                flash("Invalid email or password.", "danger")
                return redirect(url_for("login"))

            session["user_id"] = user.id

        flash("Welcome back!", "success")
        return redirect(url_for("dashboard"))

    return render_template("auth_login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You’ve been logged out.", "info")
    return redirect(url_for("login"))


# ============================================
# Core pages
# ============================================
@app.route("/")
def index():
    if current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    with get_db() as db:
        used = get_monthly_usage(db, user.id)
    return render_template(
        "dashboard.html",
        user=user,
        free_limit=FREE_PLAN_LIMIT,
        used=used,
    )


@app.route("/devotional", methods=["GET", "POST"])
@login_required
def devotional():
    user = current_user()
    output = None
    used = 0

    with get_db() as db:
        used = get_monthly_usage(db, user.id)

        if request.method == "POST":
            if not can_generate(db, user):
                flash(
                    "You’ve reached your 5 free monthly generations. "
                    "Upgrade to unlock unlimited access.",
                    "danger",
                )
                return redirect(url_for("account"))

            theme = request.form.get("theme", "").strip()
            age_range = request.form.get("age_range", "")
            num_days = request.form.get("num_days", "")
            tone = request.form.get("tone", "")
            translation = request.form.get("translation", "")

            if not theme or not age_range or not num_days or not tone:
                flash("Please fill in all required fields.", "danger")
                return redirect(url_for("devotional"))

            try:
                output = generate_kids_devotional_ai(
                    theme,
                    age_range,
                    num_days,
                    tone,
                    translation or "NIV",
                )
            except openai.RateLimitError:
                flash(
                    "We hit the AI usage limit for now. "
                    "Here’s a simple placeholder devotional instead.",
                    "warning",
                )
                output = generate_kids_devotional_stub(
                    theme,
                    age_range,
                    num_days,
                    tone,
                    translation or "NIV",
                )
            except Exception as e:
                print("Devotional AI error:", e)
                flash(
                    "Something went wrong generating with AI. "
                    "Using a simple placeholder instead.",
                    "warning",
                )
                output = generate_kids_devotional_stub(
                    theme,
                    age_range,
                    num_days,
                    tone,
                    translation or "NIV",
                )

            gen = Generation(
                user_id=user.id,
                gen_type="kids_devotional",
                theme=theme,
                input_data=json.dumps(
                    {
                        "theme": theme,
                        "age_range": age_range,
                        "num_days": num_days,
                        "tone": tone,
                        "translation": translation,
                    }
                ),
                output_data=json.dumps(output),
            )
            db.add(gen)
            db.commit()

            used += 1

    return render_template(
        "devotional.html",
        user=user,
        output=output,
        free_limit=FREE_PLAN_LIMIT,
        used=used,
    )


@app.route("/tiktok", methods=["GET", "POST"])
@login_required
def tiktok():
    user = current_user()
    output = None
    used = 0

    with get_db() as db:
        used = get_monthly_usage(db, user.id)

        if request.method == "POST":
            if not can_generate(db, user):
                flash(
                    "You’ve reached your 5 free monthly generations. "
                    "Upgrade to unlock unlimited access.",
                    "danger",
                )
                return redirect(url_for("account"))

            theme = request.form.get("theme", "").strip()
            audience = request.form.get("audience", "")
            length = request.form.get("length", "")
            tone = request.form.get("tone", "")
            translation = request.form.get("translation", "")

            if not theme or not audience or not length or not tone:
                flash("Please fill in all required fields.", "danger")
                return redirect(url_for("tiktok"))

            try:
                output = generate_tiktok_script_ai(
                    theme,
                    audience,
                    length,
                    tone,
                    translation or "NIV",
                )
            except openai.RateLimitError:
                flash(
                    "We hit the AI usage limit for now. "
                    "Here’s a simple placeholder script instead.",
                    "warning",
                )
                output = generate_tiktok_script_stub(
                    theme,
                    audience,
                    length,
                    tone,
                    translation or "NIV",
                )
            except Exception as e:
                print("TikTok AI error:", e)
                flash(
                    "Something went wrong generating with AI. "
                    "Using a simple placeholder instead.",
                    "warning",
                )
                output = generate_tiktok_script_stub(
                    theme,
                    audience,
                    length,
                    tone,
                    translation or "NIV",
                )

            gen = Generation(
                user_id=user.id,
                gen_type="tiktok_script",
                theme=theme,
                input_data=json.dumps(
                    {
                        "theme": theme,
                        "audience": audience,
                        "length": length,
                        "tone": tone,
                        "translation": translation,
                    }
                ),
                output_data=json.dumps(output),
            )
            db.add(gen)
            db.commit()

            used += 1

    return render_template(
        "tiktok.html",
        user=user,
        output=output,
        free_limit=FREE_PLAN_LIMIT,
        used=used,
    )


@app.route("/history")
@login_required
def history():
    user = current_user()
    with get_db() as db:
        gens = (
            db.query(Generation)
            .filter_by(user_id=user.id)
            .order_by(Generation.created_at.desc())
            .all()
        )
    return render_template("history.html", user=user, gens=gens)


@app.route("/account")
@login_required
def account():
    user = current_user()
    with get_db() as db:
        used = get_monthly_usage(db, user.id)
    return render_template(
        "account.html",
        user=user,
        free_limit=FREE_PLAN_LIMIT,
        used=used,
    )


@app.route("/upgrade", methods=["GET"])
@login_required
def upgrade():
    user = current_user()
    if user.plan == "creator":
        flash("You’re already on the Creator plan.", "info")
        return redirect(url_for("account"))
    return render_template("upgrade.html", user=user)


@app.route("/create-checkout-session", methods=["POST"])
@login_required
def create_checkout_session():
    user = current_user()

    if not stripe.api_key or not STRIPE_PRICE_ID:
        flash("Stripe is not configured yet. Please contact support.", "danger")
        return redirect(url_for("upgrade"))

    domain = request.host_url.rstrip("/")

    try:
        checkout_session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[
                {
                    "price": STRIPE_PRICE_ID,
                    "quantity": 1,
                }
            ],
            customer_email=user.email,
            success_url=f"{domain}{url_for('upgrade_success')}",
            cancel_url=f"{domain}{url_for('upgrade')}",
            metadata={"user_id": str(user.id)},
        )
    except Exception as e:
        print("Stripe checkout error:", e)
        flash("There was a problem starting the checkout session. Try again.", "danger")
        return redirect(url_for("upgrade"))

    return redirect(checkout_session.url)


@app.route("/upgrade-success")
@login_required
def upgrade_success():
    flash("Your subscription was successful! You’re now on the Creator plan.", "success")
    return redirect(url_for("account"))


@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    if not STRIPE_WEBHOOK_SECRET:
        return "", 400

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        return "", 400
    except stripe.error.SignatureVerificationError:
        return "", 400

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        metadata = session_obj.get("metadata", {})
        user_id_str = metadata.get("user_id")

        if user_id_str:
            with get_db() as db:
                user = db.query(User).filter_by(id=int(user_id_str)).first()
                if user:
                    user.plan = "creator"
                    if hasattr(user, "monthly_generations"):
                        user.monthly_generations = 0
                    db.commit()
                    print(f"Upgraded user {user.email} to Creator plan via Stripe.")

    return "", 200


if __name__ == "__main__":
    # Local dev server
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
