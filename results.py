from flask_server import db

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String())
    saved_path = db.Column(db.String())
    hash_value = db.Column(db.String())

    def __repr__(self):
        return '<id {}>'.format(self.id)
