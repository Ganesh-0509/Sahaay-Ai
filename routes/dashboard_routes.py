from flask import Blueprint, render_template
from flask_login import login_required, current_user
from translations.translation_utils import get_user_translations

dash_bp = Blueprint('dashboard', __name__)

@dash_bp.route('/dashboard')
@login_required
def dashboard():
    lang = get_user_translations()
    return render_template('home.html', lang=lang)

@dash_bp.route('/dashboard/mood')
@login_required
def dashboard_mood():
    return render_template('mood.html', user=current_user, lang=get_user_translations())

@dash_bp.route('/dashboard/settings')
@login_required
def dashboard_settings():
    return render_template('settings.html', user=current_user, lang=get_user_translations())

@dash_bp.route('/dashboard/analytics')
@login_required
def dashboard_analytics():
    return render_template('analytics.html', user=current_user, lang=get_user_translations())

@dash_bp.route('/dashboard/tools')
@login_required
def dashboard_tools():
    return render_template('tools.html', user=current_user, lang=get_user_translations())
