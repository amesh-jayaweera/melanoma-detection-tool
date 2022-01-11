from flask import Flask, request, jsonify
from flask_cors import CORS
from melanoma_detection_pps import detect_melanoma_by_pps, Data
from melanoma_dermoscopy import checkMelanoma

app = Flask(__name__)
cors = CORS(app, resources={r"/predict": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict/pps', methods=['POST'])
def results():
    try:
        data = request.get_json(force=True)
        pps = Data(data['age'], data['FATHMM'], data['gene'], data['tumor'], data['tier'], data['mutated dna'])
        output = detect_melanoma_by_pps(pps)
        response = jsonify(output)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200  # request completed successfully
    except Exception as e:
        print(e)
        return "Bad request", 400  # bad request

@app.route('/predict/image', methods=['POST'])
def imageResults():
    try:
        img = request.files['image']
        extension = img.filename.split(".")[1]
        if extension in required_file_extensions:
            output = checkMelanoma(img)
            response = jsonify(output)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200  # request completed successfully
        else:
            return "Bad request", 400
    except Exception as e:
        print(e)
        return "Bad request", 400  # bad request


if __name__ == "__main__":
    app.run()
