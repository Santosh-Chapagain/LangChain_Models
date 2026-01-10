from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name': 'Santosh' , 'age': '19'} #It will run code if we give age as str . No runtime error
print(new_person)