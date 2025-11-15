from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, current_user
from models.user import User
from models.forms import SignupForm

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        from app import db
        user = User.get_by_email(email, db)
        # Add password check logic here
        if user:
            login_user(user)
            
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
            
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard.dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
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
