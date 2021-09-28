from app import db

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String())
    comment = db.Column(db.String())
    createdAt = db.Column(db.Integer())

    def __repr__(self):
        return '<id {}>'.format(self.id)
