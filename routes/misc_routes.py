from flask import Blueprint, render_template, session, jsonify
from translations.translation_utils import LANGUAGES

misc_bp = Blueprint('misc', __name__)

@misc_bp.route('/')
def index():
    return render_template('index.html')

@misc_bp.route('/privacy')
def privacy():
    return render_template('privacy.html')


# Simple endpoint to set the user's language in session
@misc_bp.route('/set_language/<string:lang>', methods=['GET'])
def set_language(lang: str):
    if lang not in LANGUAGES:
        return jsonify({'ok': False, 'error': 'unsupported language'}), 400
    session['language'] = lang
    return jsonify({'ok': True, 'language': lang})
