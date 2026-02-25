from fastapi import FastAPI
import pickle
from typing import List
import numpy as np

model = pickle.load(open("heart_model.pkl", "rb"))
app = FastAPI()
@app.get("/")
def home():
    return{"Status": "running"}

@app.post("/predict")
def predict(values:list[float]):
    arr = np.asarray(values)
    arr = arr.reshape(1,-1)

    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0][1]

    return {
        "prediction": int(pred),
        "heart_disease": bool(pred),
        "confidence": float(prob)
    }
    '''
    pred = model.predict(arr)
    if pred[0] == 1:
        return {"result": "This person has heart problem"}
    else:
        return {"result": "This person does not have heart problem"}
    '''
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)