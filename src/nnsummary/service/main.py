import json
import os
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_json', methods=['POST'])
def get_json():
    person = request.form['person']
    print(f"{person.replace(' ', '-')}.json")
    with open(os.path.join("static", f"summarized_{person}.json")) as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
