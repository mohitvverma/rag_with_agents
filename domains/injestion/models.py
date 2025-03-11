from enum import Enum
from typing import Any, Literal, Optional, List, TypedDict, Dict
from pydantic import BaseModel
from domains.settings import config_settings

FILE_TYPE = [
    "pdf",
    "txt",
    "docx"
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
    namespace: Optional[str] = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE
