from pydantic import BaseModel

class Student(BaseModel):
    name: str
    

new_student = {'name': 'Santosh'}

std = Student(**new_student)
print(std) 