from flask import Flask, render_template, request
import random
import os
from gtts import gTTS

from dialogue_manager import DialogueManager
from utils import *
app = Flask(__name__, instance_relative_config=True)

dm = DialogueManager(RESOURCE_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    answer = dm.generate_answer(userText)
    return { 
            "text": answer, 
            "audioURL": getAudioURL(answer)  
        }

@app.route('/download_and_remove/<filename>')
def download_and_remove(filename):
    def generate():
        with open(filename, mode="rb") as f:
            yield from f

        os.remove(filename)

    r = app.response_class(generate(), mimetype='text/mp3')
    r.headers.set('Content-Disposition', 'attachment', filename='answer.mp3')
    return r

def getAudioURL(text): 
    path = "audio-%s.mp3" % str(random.randint(1,10000))
    gTTS(text=text, lang="en").save(path)
    return "/download_and_remove/"+path

if __name__ == "__main__":
    app.run(debug=True)
    
