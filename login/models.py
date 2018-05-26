from django.db import models


class Users(models.Model):
    name = models.CharField(max_length=50)
    age = models.CharField(max_length=20)

    def __str__(self):
        return self.name + ' - ' + self.age
