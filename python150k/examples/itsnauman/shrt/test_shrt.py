import unittest
import os

from app import db, app
from app.models import Url
from app.views import random_key

basedir = os.path.abspath(os.path.dirname(__file__))
test_db_path = 'sqlite:///' + os.path.join(basedir, 'data-test.sqlite')


class TestShrt(unittest.TestCase):

    test_url = 'http://google.com'

    def setUp(self):
        self.client = app.test_client()
        app.config['SQLALCHEMY_DATABASE_URI'] = test_db_path
        app.config['TESTING'] = True
        # Generate hash for test url
        self.key = random_key(self.test_url)
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_index(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)
        assert 'form' in r.data

    def test_flash_messages(self):
        form = {'url': 'http://abcd123hh.co.uk'}
        r = self.client.post('/form', data=form)
        self.assertEqual(r.status_code, 200)
        # Flash *Invalid Link* error incase of inactive link
        assert "Invalid Link" in r.data

    def test_404_error_handler(self):
        r = self.client.get('/abcd/abcd')
        self.assertEqual(r.status_code, 404)
        assert "404: Not Found" in r.data

    def test_form(self):
        form = {'url': self.test_url}
        r = self.client.post('/form', data=form)
        l = Url.query.filter_by(url=self.test_url).first()
        self.assertIsNotNone(l)
        self.assertEqual(l.random_code, self.key)

    def test_expand_shortened_link(self):
        link = Url(self.key, self.test_url)
        db.session.add(link)
        db.session.commit()
        # Get /<key>
        r = self.client.get(self.key)
        self.assertEqual(r.status_code, 302)
        # Redirect to expanded link
        assert self.test_url and "redirected" in r.data

    def test_expand_unshortened_link(self):
        r = self.client.get('abcd')
        self.assertEqual(r.status_code, 302)
        # Redirect to index page
        assert "/" and "redirected" in r.data

if __name__ == '__main__':
    unittest.main()
