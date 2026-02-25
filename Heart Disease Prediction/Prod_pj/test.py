from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "hello bitches"}

'''
from fastapi import FastAPI
import torch
import torch.nn as nn

app = FastAPI()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10,1)

    def forward(self,x):
        return self.fc(x)

model = MyModel()

checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint["model_state"])
model.eval()


@app.get("/")
def home():
    return {"status":"running"}


@app.post("/predict")
def predict(values:list):

    x=torch.tensor([values],dtype=torch.float32)

    y=model(x)

    return {"prediction":float(y)}
'''
'''
torch.save({
    "model_state": model.state_dict(),
    "input_size": 10
}, "model.pt")
'''
#run in terminal $uvicorn main:app --reload

'''
project/
│
├── model/
│     model.pt
│
├── src/
│     model.py
│     predict.py
│
├── main.py
│
├── requirements.txt
│
└── Dockerfile
'''