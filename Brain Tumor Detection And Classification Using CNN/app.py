import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
from database import db_connect,user_reg,user_loginact,user_upload,user_viewimages
from database import db_connect,v_image,image_info
from database import db_connect 
from werkzeug.utils import secure_filename
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
 

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route("/")
def FUN_root():
    return render_template("index.html")

@app.route("/index.html")
def logout():
    return render_template("index.html")

@app.route("/register.html")
def reg():
    return render_template("register.html")

@app.route("/login.html")
def login():
    return render_template("login.html")

@app.route("/upload.html")
def up():
    return render_template("upload.html")

@app.route("/viewdata.html")
def up1():
    return render_template("viewdata.html")

@app.route("/home.html")
def home():
    return render_template("home.html")

# -------------------------------------------register-------------------------------------------------------
@app.route("/regact", methods = ['GET','POST'])
def registeract():
   if request.method == 'POST':    
      id="0"
      status = user_reg(id,request.form['username'],request.form['password'],request.form['email'],request.form['mobile'],request.form['address'])
      if status == 1:
       return render_template("login.html",m1="sucess")
      else:
       return render_template("register.html",m1="failed")
#--------------------------------------------Login-----------------------------------------------------
@app.route("/loginact", methods=['GET', 'POST'])
def useract():
    if request.method == 'POST':
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:
            session['username'] = request.form['username']                             
            return render_template("userhome.html", m1="sucess")
        else:
            return render_template("login.html", m1="Login Failed")
#-------------------------------------------Upload Image----------------------------------
@app.route("/upload", methods = ['GET','POST'])
def upload():
   if request.method == 'POST':    
      id="0"
      status = user_upload(id,request.form['name'],request.form['image'])
      if status == 1:
       return render_template("upload.html",m1="sucess")
      else:
       return render_template("upload.html",m2="failed")
#--------------------------------------View Images-----------------------------------------
@app.route("/viewimage.html")
def viewimages():
    data = user_viewimages(session['username'])
	 
    print(data)
    return render_template("viewimage.html",user = data)

#---------------------------------------Track-----------------------------------------------
@app.route("/track")
def track():
    name = request.args.get('name')
    iname = request.args.get('iname')
    print("sdfdffsfsfdfaffdfdfsfsf")
    print(name)
    print(iname)
    data = image_info(iname)
    print("dddddddddddddddddddddddddddd")
    print(data) 
    data = v_image(data)
    print("dddddddddddddddddddddddddddd")
    print(data)
    return render_template("viewdata.html",m1="sucess",users=data,im=iname,au='augment.png',gh='graph.png')
    
 
chatbot = ChatBot(
    'CBIT-CHAT BOT',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        {
        'import_path': 'chatterbot.logic.BestMatch',
        'default_response': 'I am sorry, but I do not understand. I am still learning.',
        'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
) 
 # Training with Personal Ques & Ans 
training_data_quesans = open('C:/Users/91984/Desktop/Brain-Tumor-Detection-segmentation/health.txt').read().splitlines()
#training_data_personal = open('training_data/simple.txt').read().splitlines()
#training_data_conv = open('training_data/more.txt').read().splitlines()

training_data = training_data_quesans
print(training_data)
trainer = ListTrainer(chatbot)
trainer.train(training_data) 
# Training with English Corpus Data 
trainer_corpus = ChatterBotCorpusTrainer(chatbot)


app.static_folder = 'static'

    
# @app.route("/")
# def home():
#     return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("dddddddddddddddddddddddddddd")
    print(userText)
    return str(chatbot.get_response(userText))

# ----------------------------------------------Update Item------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000,use_reloader=False)