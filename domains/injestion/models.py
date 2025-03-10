from enum import Enum
from typing import Any, Literal, Optional, List, TypedDict, Dict
from pydantic import BaseModel


FILE_TYPE = [
    "pdf",
    "txt",
    "docx"
]

PROCESS_TYPE = Literal[
  "text",
  "image",
  "excel",
  "text_with_image",
  "audio",
  "survey_excel",
  "sql_excel",
  "video",
]

class FileInjestionResponseDto(BaseModel):
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    original_file_name: Optional[str] = None
    total_pages: Optional[int] = None

class StatusRequestDto(BaseModel):
    request_id: int
    response_data_api_path: str

class InjestRequestDto(StatusRequestDto):
    pre_signed_url: str
    file_name: str
    original_file_name: str
    file_type: str
    process_type: str
    params: Dict[str, Any]
    metadata: List[Dict[str, str]] = [{}]
    namespace: str = None

