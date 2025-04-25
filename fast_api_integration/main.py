import os
from fastapi import FastAPI

# Pull in the load_models function (and underlying `models` dict)
from fast_api_integration.models import load_models

# Import your router
from fast_api_integration.api.solve import router as solve_router

app = FastAPI(
    title="BlockBlast Solver API",
    description="FastAPI endpoints for running RL-based BlockBlast solvers",
    version="1.0.0",
)


# Hook: load all SB3 models once at startup
@app.on_event("startup")
def on_startup():
    load_models()


# Mount the /api/solve router
app.include_router(solve_router, prefix="/api")
