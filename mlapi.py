from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

class ScoringMovie(BestModel):
    Movie_Text:str


with open('model.pkl','rb') as f:
    model = pickle.load()


@app.post('/')
async def scoring_endpoint(movie:ScoringMovie):
    df =pd.DataFrame([movie.dict().values()],columns=movie.dict().keys())
    yhat = model.predict(df)
    return {"prediction:" : int(yhat)}


@app.get("/")
def read_root():
    return {"Hello": "World"}