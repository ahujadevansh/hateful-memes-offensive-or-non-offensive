import os
import urllib.request
from flask import Flask, render_template , request, flash, redirect, url_for
from prediction import *
UPLOAD_FOLDER = 'static/memes/'

app = Flask(__name__) 

app.secret_key = "ce0ba1266df2050f3c99571975fee9c802a1ce00e0012a7d"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(image):
	return '.' in image and image.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/', methods=['GET','POST'])      
def index():
    filename = 'nopic.jpg'
    if request.method == 'POST':
        image = request.files['image']
        if image and allowed_file(image.filename):
            name, ext = os.path.splitext(image.filename)
            filename = f'meme_{name}{ext}'
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_model = request.form.get('image_model')
            text_model = request.form.get('text_model')
            pred = prediction(filename, image_model, text_model)[0]
            print(f"prediction is {pred}")
            flash('Image successfully uploaded and displayed below')
            return render_template("index.html", pred = pred, is_post = True, filename = filename)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template("index.html", is_post = False, filename = 'nopic.jpg')

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='memes/' + filename), code=301)

  
if __name__=='__main__':
    app.debug = True
    app.run()
    app.run(debug = True)