# YEH AAPKA BACKEND HAI

import torch
from flask import Flask, request, send_file, jsonify, render_template
from model import Generator # model.py se Generator class import karein
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import io

# -- Model ko sirf ek baar load karein --
print("Model ko load kiya jaa raha hai... कृपया प्रतीक्षा करें।")
device = torch.device('cpu')
model_path = 'models/generator_final.pth'

# Pehle model ka khaali structure banayein
model = Generator()
# Ab usmein trained weights load karein
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # Model ko evaluation (inference) mode mein set karein

print("✅ Model loaded successfully on CPU.")

# -- Upscale Function --
def upscale_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    lr_tensor = ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_img = ToPILImage()(sr_tensor.squeeze(0))
    return sr_img

# -- Flask App --
app = Flask(__name__)

# Pehla route: HTML page dikhane ke liye
@app.route('/')
def index():
    return render_template('index.html')

# Doosra route: Image upscale karne ke liye
@app.route('/upscale', methods=['POST'])
def handle_upscale():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    
    try:
        upscaled_image = upscale_image(image_bytes)
        img_io = io.BytesIO()
        upscaled_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)