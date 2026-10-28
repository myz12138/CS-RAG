import asyncio
import json

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from backend.app.api.error_messages import http_error_handler, validation_error_handler, validation_message


def test_validation_prose_is_chinese_without_changing_field_identifiers():
    error = {"type": "greater_than_equal", "loc": ("body", "max_tokens"), "msg": "Input should be greater than or equal to 1", "ctx": {"ge": 1}}
    result = asyncio.run(validation_error_handler(None, RequestValidationError([error])))
    detail = json.loads(result.body)["detail"][0]
    assert result.status_code == 422
    assert detail == {"type": "greater_than_equal", "loc": ["body", "max_tokens"], "msg": "数值不能小于 1"}
    assert validation_message({"type": "value_error", "msg": "Neff 参数无效"}) == "Neff 参数无效"


def test_http_errors_keep_status_headers_and_custom_chinese_detail():
    result = asyncio.run(http_error_handler(None, HTTPException(404, "Not Found")))
    assert result.status_code == 404
    assert json.loads(result.body)["detail"] == "请求的资源不存在"
    result = asyncio.run(http_error_handler(None, HTTPException(401, "API Key 未配置", headers={"WWW-Authenticate": "Bearer"})))
    assert json.loads(result.body)["detail"] == "API Key 未配置"
    assert result.headers["www-authenticate"] == "Bearer"
