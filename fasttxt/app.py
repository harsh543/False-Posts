import flask
import re
from flask import Flask, request
import fasttext
import json

modelPath = "../model/modelML.bin"
model = fasttext.load_model(modelPath)

app = Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
	query = request.args.get("q")
	query = query.lower()
	query = query.strip()
	query = query.replace('\n', " ")
	query = query.replace('\t', " ")
	query = re.sub(r"[^a-zA-Z0-9]+", ' ', query)
	predictions = model.predict(query, k=5)

	# print("\n\n" + str(predictions) + "\n\n")
	predictionsToShowDict = dict()
	for label, probability in zip(predictions[0], predictions[1]):
		label = label.replace("__label__", "")
		predictionsToShowDict[label] = probability

	print(predictionsToShowDict)
	return json.dumps(predictionsToShowDict)


if __name__ == "__main__":
	app.run()


