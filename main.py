import eel
from numpy import block

id = 0
username = ''

eel.init('web')
@eel.expose
def detect():
    import face_recognition


@eel.expose
def getdata():
    import face_dataset

@eel.expose
def getValue(a, b):
   global id
   id = a
   global username
   username = b

   print(str(id) + str(username))

eel.start('index.html')

