"""Localize framework error prose without changing API codes or algorithm data."""

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException


def validation_message(error: dict[str, Any]) -> str:
    message = str(error.get("msg", ""))
    if any("\u3400" <= char <= "\u9fff" for char in message):
        return message
    ctx = error.get("ctx") or {}
    labels = {
        "missing": "此项为必填项", "string_type": "请输入文本",
        "int_type": "请输入整数", "int_parsing": "请输入有效整数",
        "float_type": "请输入数值", "float_parsing": "请输入有效数值",
        "bool_parsing": "请选择有效的开关值", "literal_error": "请选择允许的选项",
        "json_invalid": "请求 JSON 格式不正确", "extra_forbidden": "包含不支持的参数",
        "string_too_short": f"文本长度至少为 {ctx.get('min_length', '规定数量')} 个字符",
        "string_too_long": f"文本长度最多为 {ctx.get('max_length', '规定数量')} 个字符",
        "greater_than_equal": f"数值不能小于 {ctx.get('ge', '规定下限')}",
        "less_than_equal": f"数值不能大于 {ctx.get('le', '规定上限')}",
        "greater_than": f"数值必须大于 {ctx.get('gt', '规定下限')}",
        "less_than": f"数值必须小于 {ctx.get('lt', '规定上限')}",
    }
    return labels.get(error.get("type"), "参数格式不正确")


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(error.get("loc", [])), "type": error.get("type"), "msg": validation_message(error)}
        for error in exc.errors()
    ]})


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    translations = {"Not Found": "请求的资源不存在", "Method Not Allowed": "接口不支持此操作",
                    "Unauthorized": "身份验证失败", "Forbidden": "没有访问权限",
                    "Internal Server Error": "服务内部异常，请查看后端日志"}
    detail = translations.get(exc.detail, exc.detail) if isinstance(exc.detail, str) else exc.detail
    return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers)
