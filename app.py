import os
import stripe
import json
from datetime import datetime
from functools import wraps

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

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FAITHFLOW_SECRET_KEY", "dev-secret-change-me")

# Stripe configuration (fill these from your Stripe Dashboard)
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")  # e.g. price_12345 from Stripe
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


# OpenAI client (reads OPENAI_API_KEY from environment)
client = OpenAI()

# -----------------------------
# Database setup (SQLite)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "faithflow.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Plan / usage
    plan = Column(String(50), default="free")  # "free" or "creator"
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

# -----------------------------
# Helpers
# -----------------------------

FREE_PLAN_LIMIT = 5  # Free users get 5 generations/month (dev+TikTok combined)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None

    db = next(get_db())
    user = db.query(User).filter_by(id=user_id).first()

    # If the user no longer exists in the DB (e.g., DB recreated),
    # clear the session so we don't get stuck in a weird state.
    if not user:
        session.clear()
        return None

    return user



def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def can_generate(user: User) -> bool:
    """Return True if user is allowed to generate more content."""
    if user.plan == "creator":
        return True
    return (user.monthly_generations or 0) < FREE_PLAN_LIMIT


# -----------------------------
# Stub generators (fallbacks)
# -----------------------------
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


# -----------------------------
# AI generators
# -----------------------------
def generate_kids_devotional_ai(theme, age_range, num_days, tone, translation):
    """
    Use OpenAI to generate a structured kids devotional plan.
    Returns a list of day dicts matching the template expectations.
    """
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
    // one object per day up to {days_count}
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

    # Strip ```json ... ``` style fences if the model added them
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
        # Fallback: show raw output as a single day so you don't lose data
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
    """
    Use OpenAI to generate a structured TikTok Bible script.
    Returns a dict with:
      hook, script_lines, verses[{reference, text, note}], cta, caption.
    """
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
- Start with a strong hook that speaks directly to someone experiencing "{theme}".
- Use short, natural lines that can be spoken on camera.
- Include 1–2 Bible verses in the requested translation, plus a simple explanation for each.
- End with a hopeful, Christ-centered takeaway.
- Include a simple call to action (e.g., save/share/pray).
- Include a caption with 5–10 relevant hashtags (no spammy ones like #viral).

Return a single JSON object with this exact structure:

{{
  "hook": "Short opening hook line.",
  "script_lines": [
    "First spoken line.",
    "Second spoken line.",
    "..."
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
- Do not include any keys other than hook, script_lines, verses, cta, caption.
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

    # Strip ```json ... ``` style fences if present
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
        # Fallback: treat as plain text script
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

    output = {
        "hook": data.get("hook", ""),
        "script_lines": data.get("script_lines", []),
        "verses": verses,
        "cta": data.get("cta", ""),
        "caption": data.get("caption", ""),
        "length": length,
        "tone": tone,
    }
    return output


# -----------------------------
# Auth routes
# -----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("signup"))

        db = next(get_db())
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
    # If already logged in, go straight to dashboard
    if current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Debug: print what we got from the form
        print("LOGIN attempt:", repr(email))

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return redirect(url_for("login"))

        db = next(get_db())
        user = db.query(User).filter_by(email=email).first()

        if not user:
            flash("No account found with that email.", "danger")
            return redirect(url_for("login"))

        if not check_password_hash(user.password_hash, password):
            flash("Incorrect password. Please try again.", "danger")
            return redirect(url_for("login"))

        # Success
        session["user_id"] = user.id
        flash("Welcome back!", "success")
        return redirect(url_for("dashboard"))

    # GET request
    return render_template("auth_login.html")



@app.route("/logout")
def logout():
    session.clear()
    flash("You’ve been logged out.", "info")
    return redirect(url_for("login"))


# -----------------------------
# Core pages
# -----------------------------
@app.route("/")
def index():
    if current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    return render_template("dashboard.html", user=user)


@app.route("/devotional", methods=["GET", "POST"])
@login_required
def devotional():
    user = current_user()
    output = None

    if request.method == "POST":
        # Free plan limit check
        if not can_generate(user):
            flash("You’ve reached your monthly free limit. Upgrade to generate more content.", "danger")
            return redirect(url_for("account"))

        theme = request.form.get("theme", "").strip()
        age_range = request.form.get("age_range", "")
        num_days = request.form.get("num_days", "")
        tone = request.form.get("tone", "")
        translation = request.form.get("translation", "")

        if not theme or not age_range or not num_days or not tone:
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("devotional"))

                # AI with graceful fallback to stub
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
            flash("Something went wrong generating with AI. Using a simple placeholder instead.", "warning")
            output = generate_kids_devotional_stub(
                theme,
                age_range,
                num_days,
                tone,
                translation or "NIV",
            )

        # Use the SAME DB session for both user update and Generation insert
        db = next(get_db())

        # Re-load the user in this session
        db_user = db.query(User).filter_by(id=user.id).first()

        # Safely bump free-plan usage
        if db_user and db_user.plan == "free":
            db_user.monthly_generations = (db_user.monthly_generations or 0) + 1

        gen = Generation(
            user_id=db_user.id if db_user else user.id,
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


    return render_template("devotional.html", user=user, output=output)


@app.route("/tiktok", methods=["GET", "POST"])
@login_required
def tiktok():
    user = current_user()
    output = None

    if request.method == "POST":
        # Free plan limit check
        if not can_generate(user):
            flash("You’ve reached your monthly free limit. Upgrade to generate more content.", "danger")
            return redirect(url_for("account"))

        theme = request.form.get("theme", "").strip()
        audience = request.form.get("audience", "")
        length = request.form.get("length", "")
        tone = request.form.get("tone", "")
        translation = request.form.get("translation", "")

        if not theme or not audience or not length or not tone:
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("tiktok"))

                # AI with graceful fallback
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
            flash("Something went wrong generating with AI. Using a simple placeholder instead.", "warning")
            output = generate_tiktok_script_stub(
                theme,
                audience,
                length,
                tone,
                translation or "NIV",
            )

        db = next(get_db())
        db_user = db.query(User).filter_by(id=user.id).first()

        if db_user and db_user.plan == "free":
            db_user.monthly_generations = (db_user.monthly_generations or 0) + 1

        gen = Generation(
            user_id=db_user.id if db_user else user.id,
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


    return render_template("tiktok.html", user=user, output=output)


@app.route("/history")
@login_required
def history():
    user = current_user()
    db = next(get_db())
    gens = (
        db.query(Generation)
        .filter_by(user_id=user.id)
        .order_by(Generation.created_at.desc())
        .all()
    )
    return render_template("history.html", user=user, gens=gens)

@app.route("/debug-users")
def debug_users():
    db = next(get_db())
    users = db.query(User).order_by(User.id.asc()).all()
    rows = []
    for u in users:
        rows.append(f"{u.id} • {u.email} • plan={u.plan} • gens={u.monthly_generations}")
    if not rows:
        return "No users found in DB."
    return "<br>".join(rows)


@app.route("/account")
@login_required
def account():
    user = current_user()
    return render_template("account.html", user=user, free_limit=FREE_PLAN_LIMIT)

@app.route("/upgrade", methods=["GET"])
@login_required
def upgrade():
    user = current_user()
    # If already creator, just send to account
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

    # Your app's domain – adjust if you deploy
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
            metadata={
                "user_id": str(user.id),
            },
        )
    except Exception as e:
        print("Stripe checkout error:", e)
        flash("There was a problem starting the checkout session. Try again.", "danger")
        return redirect(url_for("upgrade"))

    return redirect(checkout_session.url)

@app.route("/upgrade-success")
@login_required
def upgrade_success():
    flash("If your payment was successful, your plan will update shortly.", "success")
    return redirect(url_for("account"))


@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    if not STRIPE_WEBHOOK_SECRET:
        # For safety, don't process webhooks without a secret configured
        return "", 400

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        # Invalid payload
        return "", 400
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        return "", 400

    # Handle the event
    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        metadata = session_obj.get("metadata", {})
        user_id_str = metadata.get("user_id")

        if user_id_str:
            db = next(get_db())
            user = db.query(User).filter_by(id=int(user_id_str)).first()
            if user:
                user.plan = "creator"
                # Optionally reset usage
                user.monthly_generations = 0
                db.commit()
                print(f"Upgraded user {user.email} to Creator plan via Stripe.")

    # You can handle subscription events too, if needed
    return "", 200

if __name__ == "__main__":
    app.run(debug=True)
