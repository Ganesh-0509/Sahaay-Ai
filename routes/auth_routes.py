from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, current_user
from models.user import User
from models.forms import SignupForm

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        from app import db
        user = User.get_by_email(email, db)
        # Add password check logic here
        if user:
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = SignupForm(request.form)
    if request.method == 'POST' and form.validate():
        email = form.email.data
        name = form.name.data
        password = form.password.data
        consent = form.consent.data
        if not consent:
            flash('You must agree to the privacy policy to sign up.', 'danger')
            return render_template('signup.html', form=form)
        from app import db
        if User.get_by_email(email, db):
            flash('Email address already exists.', 'danger')
        else:
            new_user = User.create(email, name, password)
            login_user(new_user)
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard'))
    return render_template('signup.html', form=form)

@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))
