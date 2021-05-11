import keras
import re
import numpy as np
import pytesseract
import cv2
import pickle


from nltk.corpus import stopwords

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras import preprocessing, Input

from PIL import Image, ImageFile

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ahuja\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
EMAIL = re.compile('^([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})$')
STOPWORDS = set(stopwords.words('english'))
maxlen = 1000

models = {
    'BILSTM_RESNET': load_model(f'models//BILSTM_RESNET.h5'),
    'CNN_RESNET': load_model(f'models//CNN_RESNET.h5'),
    'Stack_LSTM_RESNET': load_model(f'models//Stack_LSTM_RESNET.h5'),
    'BILSTM_VGG': load_model(f'models//BILSTM_VGG.h5'),
    'CNN_VGG': load_model(f'models//CNN_VGG.h5'),
    'Stack_LSTM_VGG': load_model(f'models//Stack_LSTM_VGG.h5'),
}

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    text = text.lower()
    text = EMAIL.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ',text)
    text = BAD_SYMBOLS_RE.sub('',text)    
    text = text.replace('x','')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    
    return text

# Takes in image and preprocess it
def process_input(filename):

    input_path = f'static/memes/{filename}'
    # Loading image from given path and resizing it to 224*224*3 format
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = image.load_img(input_path, target_size=(224,224))
    # Converting image to array    
    img_data = image.img_to_array(img)
    # Adding one more dimension to array    
    img_data = np.expand_dims(img_data, axis=0)
    input_img = preprocess_input(img_data)

    # Extracting text from image
    # Grayscale, Gaussian blur, Otsu's threshold
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Perform text extraction
    text = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    text = [clean_text(text)]

    # converting text to sequence
    sequences_predict = tokenizer.texts_to_sequences(text)
    x_predict = preprocessing.sequence.pad_sequences(sequences_predict, maxlen=maxlen)
    # input_txt = np.expand_dims(x_predict, axis=0)
    return ([input_img, x_predict], np.array([1]))

def prediction(filename, image_model, text_model):

    input = process_input(filename)
    model = f"{text_model}_{image_model}"
    model = models[model]
    pred = model.predict(iter(input), steps=1)
    pred = np.round(pred)
    new_pred = []
    for i in pred:
        new_pred.append(np.argmax(i == 1))
    
    return new_pred









