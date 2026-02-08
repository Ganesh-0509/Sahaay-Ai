from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, current_user
from models.user import User
from models.forms import SignupForm

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    # Clear any lingering flash messages on GET request
    if request.method == 'GET':
        session.pop('_flashes', None)
    
    if request.method == 'POST':
        # Handle both JSON (from Next.js) and form data (from Jinja templates)
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
            print(f"🔍 DEBUG /login (JSON): email={email}")
        else:
            email = request.form.get('email')
            password = request.form.get('password')
            print(f"🔍 DEBUG /login (FORM): email={email}")
        
        from app import db
        user = User.get_by_email(email, db)
        
        if user and user.check_password(password):
            login_user(user)
            print(f"✅ Login successful! User ID: {user.id}")
            print(f"   Session after login_user: {dict(session)}")
            print(f"   Current user after login_user: {current_user}, authenticated: {current_user.is_authenticated}")
            
            # Load user's language preference from Firestore
            try:
                user_doc = db.collection('users').document(user.id).get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    user_lang = user_data.get('language', 'en')
                    session['language'] = user_lang
            except Exception as e:
                print(f"Failed to load user language: {e}")
                session['language'] = 'en'
            
            # Return JSON for API clients (Next.js)
            if request.is_json:
                from flask import jsonify
                return jsonify({
                    'ok': True,
                    'message': 'Logged in successfully!',
                    'user': {
                        'user_id': user.id,
                        'username': user.username,
                        'email': user.email
                    }
                }), 200
            
            # Traditional redirect for Jinja templates
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard.dashboard'))
        else:
            print(f"❌ Login failed for email: {email}")
            
            # Return JSON error for API clients
            if request.is_json:
                from flask import jsonify
                return jsonify({
                    'ok': False,
                    'message': 'Invalid email or password.'
                }), 401
            
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    # Clear any lingering flash messages on GET request
    if request.method == 'GET':
        session.pop('_flashes', None)
    
    # Handle JSON requests from Next.js
    if request.is_json:
        from flask import jsonify
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')
        password = data.get('password')
        consent = data.get('consent')
        
        print(f"🔍 DEBUG /signup (JSON): email={email}, name={name}")
        
        if not consent:
            return jsonify({'ok': False, 'message': 'You must agree to the privacy policy.'}), 400
        
        from app import db
        if not db:
            return jsonify({'ok': False, 'message': 'Database connection failed.'}), 500
        
        if User.get_by_email(email, db):
            return jsonify({'ok': False, 'message': 'Email address already exists.'}), 409
        
        new_user = User.create(email, name, password, db)
        if not new_user:
            return jsonify({'ok': False, 'message': 'Account creation failed.'}), 500
        
        login_user(new_user)
        session['language'] = 'en'
        
        print(f"✅ Signup successful! User ID: {new_user.id}")
        
        return jsonify({
            'ok': True,
            'message': 'Account created successfully!',
            'user': {
                'user_id': new_user.id,
                'username': new_user.username,
                'email': new_user.email
            }
        }), 201
    
    # Handle form data from Jinja templates
    form = SignupForm(request.form)
    if request.method == 'POST':
        if not form.validate():
            # Flash validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"{field}: {error}", 'danger')
            return render_template('signup.html', form=form)
        
        email = form.email.data
        name = form.name.data
        password = form.password.data
        consent = form.consent.data
        
        if not consent:
            flash('You must agree to the privacy policy to sign up.', 'danger')
            return render_template('signup.html', form=form)
        
        from app import db
        if not db:
            flash('Database connection failed. Please try again later.', 'danger')
            return render_template('signup.html', form=form)
        
        if User.get_by_email(email, db):
            flash('Email address already exists.', 'danger')
            return render_template('signup.html', form=form)
        else:
            new_user = User.create(email, name, password, db)
            if not new_user:
                flash('Account creation failed. Please try again.', 'danger')
                return render_template('signup.html', form=form)
            login_user(new_user)
            
            # Set default language in session for new users
            session['language'] = 'en'
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard.dashboard'))
    return render_template('signup.html', form=form)

@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
