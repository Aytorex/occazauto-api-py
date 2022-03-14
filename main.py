from flask import Flask, request, jsonify, render_template, redirect, sessions, url_for, flash, abort, Blueprint, \
    Response
from flask_dropzone import Dropzone
from training import predictannouce

app = Flask(__name__)
app.config["DEBUG"] = True  # option debug
dropzone = Dropzone(app)


@app.route('/integrity', methods=["POST"])
def integrity():
    manufacturer_name = request.json["manufacturer_name"]
    model_name = request.json["model_name"]
    transmission = request.json["transmission"]
    color = request.json["color"]
    odometer_value = request.json["odometer_value"]
    year = request.json["year"]
    engine_fuel = request.json["engine_fuel"]
    engine_type = request.json["engine_type"]
    price = request.json["price"]
    return jsonify({"trust_score": predictannouce(manufacturer_name, model_name, transmission, color, odometer_value,
                                                  year, engine_fuel, engine_type, price)})


app.run(host="127.0.0.1", port=5000, threaded=True)
