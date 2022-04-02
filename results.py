from app import db

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String())
    img_name = db.Column(db.String())
    count = db.Column(db.Integer)

    db.Index('idx_text', 'text')

    def __repr__(self):
        return '<id {}>'.format(self.id)
