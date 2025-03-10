from domains.injestion.doc_loader import file_loader
from domains.injestion.models import InjestRequestDto, FileInjestionResponseDto
from domains.models import RequestStatus, ApiNameEnum, RequestStatusEnum
from domains.injestion.utils import update_status
from domains.injestion.vector_db_utils import push_to_database
from domains.settings import config_settings
from domains.status_util import call_update_status_api

from loguru import logger

from fastapi import APIRouter, BackgroundTasks, HTTPException, Header

router = APIRouter(tags=["injestion"])


@router.post(
    "/injest-doc",
    summary="Injests a document into the database",
    description="Injests a document into the database",
)
def injest_doc(
    request: InjestRequestDto,
    background_tasks: BackgroundTasks,
    # token: str = Header(alias="Authorization"),
) -> FileInjestionResponseDto:
    logger.info(f"injest-doc request: {request.model_dump_json()}")
    try:
        logger.info(f"Extracting metadata from File")

        response = FileInjestionResponseDto(
            file_path=request.pre_signed_url,
            file_name=request.file_name,
            original_file_name=request.original_file_name,
            total_pages=1
        )

        background_tasks.add_task(
            load_file_push_to_db,
            request,
        )
        return response

    except Exception as e:
        logger.exception("Failed during fetching metadata")
        status = RequestStatus(
            request_id=request.request_id,
            api_name=ApiNameEnum.INJEST_DOC,
            status=RequestStatusEnum.FAILED,
            error_detail=str(e),
        )

        update_status(request.response_data_api_path, status)

        response = FileInjestionResponseDto(
            file_name=request.file_name,
            original_file_name=request.original_file_name,
            total_pages=0,
        )

    background_tasks.add_task(
        load_file_push_to_db, request
    )
    update_status(request.response_data_api_path, status)

    logger.info(f"injest-doc response: {response.model_dump_json()}")
    return response


def load_file_push_to_db(
        request: InjestRequestDto
):
    try:
        logger.debug(f"load_file_push_to_db(): Attempting to load file from {request.pre_signed_url}")

        chunked_documents, non_chunked_docs = file_loader(
            pre_signed_url=request.pre_signed_url,
            file_name=request.file_name,
            original_file_name=request.file_name,
            file_type=request.file_type,
            process_type=request.file_type,
            params={"summary": False},
            metadata=[]
        )
        logger.info(f"Successfully loaded file from {request.pre_signed_url} and total pages in file is {len(non_chunked_docs)}")

        push_to_database(
            texts=chunked_documents,
            index_name=config_settings.PINECONE_INDEX_NAME,
            namespace=request.namespace
        )

        # Create success status
        status = RequestStatus(
            request_id=request.request_id,
            api_name=ApiNameEnum.INJEST_DOC,
            status=RequestStatusEnum.COMPLETED,
            data_json={"total_pages": len(non_chunked_docs)},
        )
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Failed to load file from {request.pre_signed_url} and error is {e}")
        error_detail = f"Failed when process_type is {request.process_type}: {e}"
        status = RequestStatus(
            request_id=request.request_id,
            api_name=ApiNameEnum.INJEST_DOC,
            status=RequestStatusEnum.FAILED,
            error_detail=error_detail,
        )

    finally:
        if status:
            logger.info(
                f"Completed injest-doc for file_name: {request.file_name}"
                f" with status: {status.status}"
            )
            call_update_status_api(request.response_data_api_path, status)
        else:
            logger.error("Status object was not created - this is unexpected")
