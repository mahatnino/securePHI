#app.py
from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pathlib import Path
import pytesseract 
import cv2
from PIL import Image
from pytesseract import Output
import pandas as pd
import numpy as np
import json
 
app = Flask(__name__)
CORS(app)

app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_conversion(img):
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\subash.mahat\AppData\Local\Tesseract-OCR\tesseract.exe'
    img_file = img
    file_name = img_file.split('/')[2]
    img = cv2.imread(img_file)
    #gray image 
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Converting grey image to binary image by Thresholding
    thresh_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # configuring parameters for tesseract
    custom_config = r'--oem 3 --psm 6'
 
    # Get all OCR output information from pytesseract
    ocr_output_details = pytesseract.image_to_data(thresh_img,output_type=Output.DICT, config=custom_config, lang='eng')
    df = pd.DataFrame.from_dict(ocr_output_details)
    df['text'].replace('', np.nan, inplace = True)
    df.dropna(subset=['text'], inplace = True)
    for i in df.index:
        (x, y, w, h) = (df['left'][i], df['top'][i], df['width'][i], df['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 10)
    cv2.imwrite('static/uploads/'+file_name,img)
       
@app.route("/")
def index():
    return 'Homepage'

@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status

    files = request.files.getlist('files[]')
     
    errors = {}
    success = False
     
    for file in files:      
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
    
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        image_conversion('static/uploads/' + filename)
        resp = jsonify( {'message' :'Files successfully uploaded : '+ filename  }  )
        resp.status_code = 201
        #json.dumps(True)
        #app.response_class(json.dumps(True), content_type='application/json')
        return jsonify({'message' : 'Files successfully uploaded', 'ok': bool(1) })
    else:
        resp = jsonify(errors)
        #resp.status_
        return jsonify({'message' : 'Files failed to uploaded', 'ok': bool(0) })


@app.route('/uploadReturn/<filename>', methods=['GET'])
def uploadReturn_file(filename):
    my_file = Path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    if my_file.exists():
        resp = jsonify({'message' : 'File Exists : '+ filename  }  )
        resp.status_code = 200
        return jsonify({'response_code' : resp.status_code ,'path' : os.path.join(app.config['UPLOAD_FOLDER'], filename)}) 
    else: 
        resp = jsonify({'message' : 'File do not exist : '+ filename  }  )
        resp.status_code = 400
        return jsonify({'response_code' : resp.status_code ,'message' : 'File does not exists' }) 
           
if __name__ == '__main__':
    app.run(debug=True)
