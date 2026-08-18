import config
import torch
import flask
from flask import Flask
from flask import request
from modeling import BERTBaseUncased

import torch.nn as nn

app = Flask(__name__)

MODEL = None
DEVICE = "cpu"

def sentence_prediction(sentence, model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer(
        review,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get("comment")
    positive_prediction = sentence_prediction(sentence, model=MODEL)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        'toxic_probability': str(positive_prediction),
        'non_toxic_probability': str(negative_prediction),
        'sentence': str(sentence)
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL = nn.DataParallel(MODEL)
    if torch.cuda.is_available():
      map_location = lambda storage, loc: storage.cuda()
    else:
      map_location = 'cpu'

    checkpoint = torch.load(config.MODEL_PATH, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
    MODEL.module.load_state_dict(state_dict)
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()
