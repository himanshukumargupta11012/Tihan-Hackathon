from flask import Flask, render_template, request
import model
import cv2
import numpy as np
from PIL import Image
import mimetypes
import os


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/send_image', methods=['POST'])
def get_image():
    if os.path.exists('static/output.webm'):
        print("fdsfsd")
        os.remove('static/output.webm')
    if 'image' in request.files:
        image = request.files['image']
        if image:
            mime_type, _ = mimetypes.guess_type(image.filename)
            if mime_type and mime_type.startswith('video'):
                image.save('static/input.mp4')
                model.process_video('static/input.mp4')
                command = f'ffmpeg -i {"static/output.mp4"} -c:v libvpx-vp9 -c:a libvorbis {"static/output.webm"}'
                os.system(command)
                return render_template('output.html', type='video', image_url='static/output.webm')

            elif mime_type and mime_type.startswith('image'):
                rgb_arr = np.array(Image.open(image))
                # image_arr = image_arr[:, :, ::-1]
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                # image_arr = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
                model.process_image(bgr_arr)

                return render_template('output.html', type='image', image_url='static/output.png')
            
            else:
                return "Unsupported type"
        else:
            return "image not found"
    
    else:
        return "image not found"

if __name__ == "__main__":
    app.run(debug=True)


