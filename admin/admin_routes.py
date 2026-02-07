# admin/admin_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from .admin_auth import validate_admin
import datetime
import os

# Create blueprint with proper template and static folders
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')
admin_bp = Blueprint('admin', __name__, url_prefix='', template_folder=template_dir, static_folder=static_dir)

__all__ = ['admin_bp']

# Helper function to get Firestore db (lazy import to avoid circular dependency)
def get_db():
    from app import db
    return db

@admin_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if validate_admin(email, password):
            session['admin_logged_in'] = True
            return redirect(url_for('admin.admin_dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('admin/login.html')

@admin_bp.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.admin_login'))

    db = get_db()
    users_ref = db.collection('users')
    checkins_ref = db.collection_group('checkins')

    users = list(users_ref.stream())
    checkins = list(checkins_ref.stream())
    total_users = len(users)
    total_checkins = len(checkins)

    today = datetime.datetime.now(datetime.timezone.utc).date()
    sentiments = []
    moods = {}
    latest_checkins = []

    for c in checkins:
        data = c.to_dict()
        data['id'] = c.id
        
        # Handle timestamp/created_at fields
        timestamp = data.get('timestamp') or data.get('created_at')
        if timestamp and hasattr(timestamp, 'date'):
            dt = timestamp.date()
        else:
            dt = None
            
        # Collect today's sentiments and moods
        if dt == today:
            if 'sentiment' in data:
                sentiments.append(data.get('sentiment', 0))
            if 'avg_sentiment' in data:
                sentiments.append(data.get('avg_sentiment', 0))
            mood = data.get('mood') or data.get('mood_dominant')
            if mood:
                moods[mood] = moods.get(mood, 0) + 1
        
        latest_checkins.append(data)

    avg_sentiment = round(sum(sentiments)/len(sentiments), 2) if sentiments else 0
    dominant_mood = max(moods, key=moods.get) if moods else 'N/A'
    
    # Sort by timestamp, handling None values
    def get_sort_key(x):
        ts = x.get('timestamp') or x.get('created_at')
        if ts and isinstance(ts, datetime.datetime):
            return ts
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    
    latest_checkins = sorted(latest_checkins, key=get_sort_key, reverse=True)[:10]

    mood_dist = moods
    sentiment_trend = sentiments

    return render_template('admin/dashboard.html',
        total_users=total_users,
        total_checkins=total_checkins,
        avg_sentiment=avg_sentiment,
        dominant_mood=dominant_mood,
        mood_dist=mood_dist,
        sentiment_trend=sentiment_trend,
        latest_checkins=latest_checkins)

@admin_bp.route('/admin/users')
def admin_users():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.admin_login'))

    db = get_db()
    query = request.args.get('q', '').lower()
    lang = request.args.get('lang')
    users_ref = db.collection('users')
    users = []
    for u in users_ref.stream():
        data = u.to_dict()
        data['id'] = u.id
        if query and (query not in data.get('name', '').lower() and query not in data.get('email', '').lower()):
            continue
        if lang and data.get('language') != lang:
            continue
        users.append(data)
    
    # Sort by created_at, handling None values
    def get_user_sort_key(x):
        created_at = x.get('created_at')
        if created_at and isinstance(created_at, datetime.datetime):
            return created_at
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    
    users = sorted(users, key=get_user_sort_key, reverse=True)
    return render_template('admin/users.html', users=users)

@admin_bp.route('/admin/user/<user_id>')
def admin_user_detail(user_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.admin_login'))

    db = get_db()
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        flash('User not found', 'danger')
        return redirect(url_for('admin.admin_users'))
    user = user_doc.to_dict()
    user['id'] = user_id

    checkins_ref = user_ref.collection('checkins')
    checkins = [c.to_dict() for c in checkins_ref.stream()]
    conversations_ref = user_ref.collection('conversations')
    conversations = [c.to_dict() for c in conversations_ref.stream()]

    mood_list = [c.get('mood') for c in checkins if 'mood' in c]
    sentiment_trend = [c.get('sentiment') for c in checkins if 'sentiment' in c]

    return render_template('admin/user_detail.html',
        user=user,
        checkins=checkins,
        conversations=conversations,
        mood_list=mood_list,
        sentiment_trend=sentiment_trend)

@admin_bp.route('/admin/delete_user/<user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.admin_login'))

    db = get_db()
    db.collection('users').document(user_id).delete()
    flash('User deleted', 'success')
    return redirect(url_for('admin.admin_users'))

@admin_bp.route('/admin/delete_checkin/<user_id>/<checkin_id>', methods=['POST'])
def admin_delete_checkin(user_id, checkin_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.admin_login'))

    db = get_db()
    db.collection('users').document(user_id).collection('checkins').document(checkin_id).delete()
    flash('Check-in deleted', 'success')
    return redirect(url_for('admin.admin_user_detail', user_id=user_id))

@admin_bp.route('/admin/delete_conversation/<user_id>/<conv_id>', methods=['POST'])
def admin_delete_conversation(user_id, conv_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.admin_login'))

    db = get_db()
    db.collection('users').document(user_id).collection('conversations').document(conv_id).delete()
    flash('Conversation deleted', 'success')
    return redirect(url_for('admin.admin_user_detail', user_id=user_id))
