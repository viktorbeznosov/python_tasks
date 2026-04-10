# app/core/index/route.py

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from ...config import settings

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """
    Главная страница приложения
    """
    log = request.app.state.log
    await log.log_info(target="index_page", message=f"Главная страница запрошена от {request.client.host}")

    # Получаем окружение Jinja2 из app.state (вместо templates)
    jinja_env = request.app.state.jinja_env
    
    # Рендерим шаблон напрямую
    template = jinja_env.get_template("index.html")
    html_content = template.render(
        request=request,
        API_URL=str(settings.API_URL)
    )
    
    return HTMLResponse(content=html_content)


@router.get("/get-token")
async def get_token(request: Request):
    """
    Server-side endpoint для выдачи JWT frontend JS.
    Лениво получает токен через JWTClient.
    """
    log = request.app.state.log
    try:
        jwt_client = request.app.state.jwt_client
        token = await jwt_client.get_token()
        await log.log_info(target="get_token", message=f"JWT выдан от {request.client.host}")
        return JSONResponse({"access_token": token})
    except Exception as e:
        await log.log_error(target="get_token", message=f"Ошибка при выдаче JWT от {request.client.host}: {str(e)}")
        raise