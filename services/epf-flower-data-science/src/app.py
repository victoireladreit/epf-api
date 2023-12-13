from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api.router import router
from fastapi.responses import RedirectResponse


def get_application() -> FastAPI:
    application = FastAPI(
        title="epf-flower-data-science",
        description="""Fast API""",
        version="1.0.0",
        redoc_url=None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    # step3: redirection to doc swagger (UI)
    @application.get("/")
    def redirect_to_doc():
        return RedirectResponse(url="/docs")
    return application
