from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

basedir = os.path.abspath(os.path.dirname(__file__))
db_path = 'sqlite:///' + os.path.join(basedir, 'data-dev.sqlite')

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or "secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DB_URI') or db_path
app.config['BASE_LINK'] = "http://localhost:5000/"

db = SQLAlchemy(app)

from . import models
from . import views
