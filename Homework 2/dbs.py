from mongoengine import *

class Movie(Document):
    movie_id = IntField(required=True)
    name = StringField(required=True)
    released = BooleanField(required=True)
    year = IntField(required=True)
    score = FloatField(required=True)
    rating_count = IntField(required=True)
    director = StringField(required=True)


    