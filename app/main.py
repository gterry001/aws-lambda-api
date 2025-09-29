# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from logic import run_analysis
from mangum import Mangum

app = FastAPI(title="Portfolio Risk API")

# CORS abierto para pruebas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpeta de estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/run-analysis")
def run_analysis_endpoint():
    result = run_analysis()
    return {
        "ordenes": result["ordenes"],
        "grafico_url": f"/static/{result['plot_file']}"
    }
@app.get("/")
def root():
    return {"message": "API is running"}
handler = Mangum(app)


