from wtforms import Form, StringField, PasswordField, BooleanField, validators

class SignupForm(Form):
    name = StringField('Full Name', [validators.DataRequired(), validators.Length(min=2, max=50)])
    email = StringField('Email Address', [validators.DataRequired(), validators.Email()])
    password = PasswordField('Password', [validators.DataRequired(), validators.Length(min=6)])
    consent = BooleanField('Consent', [validators.DataRequired()])
