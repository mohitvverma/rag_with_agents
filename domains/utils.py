import fastapi

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from domains.settings import config_settings
from domains.retreival.chat_handler import StreamingLLMCallbackHandler
from loguru import logger

def get_chat_model(model_key="OPENAI_CHAT"):
    try:
        if config_settings.LLM_SERVICE == "openai":
            return ChatOpenAI(
                model=config_settings.LLMS.get(model_key, ""),
                temperature=0.0,
                api_key=config_settings.OPENAI_API_KEY,
            )

        elif config_settings.LLM_SERVICE == "azure_openai":
            return ChatOpenAI(
                model=config_settings.LLMS.get("SUMMARIZE_LLM_MODEL"),
                temperature=0.0,
                api_key=config_settings.OPENAI_API_KEY,
            )

    except Exception as e:
        logger.error(f"Error while getting chat model: {e}")
        return None



def get_chat_model_streaming(model_key: str ="OPENAI_CHAT"):
    if config_settings.LLM_SERVICE == "openai":
        return ChatOpenAI(
            model=config_settings.LLMS.get("OPENAI_CHAT_MODEL_NAME"),
            temperature=0.0,
            api_key=config_settings.OPENAI_API_KEY,
            streaming=True
        )

    elif config_settings.LLM_SERVICE == "azure_openai":
        return ChatOpenAI(
            model=config_settings.LLMS.get("SUMMARIZE_LLM_MODEL"),
            temperature=0.0,
            api_key=config_settings.OPENAI_API_KEY,
            streaming=True
        )


def get_chat_model_with_streaming(
    websocket: fastapi.WebSocket,
    model_key: str = "OPENAI_CHAT",
    temperature: float = 0.0,
):
    try:
        if config_settings.LLM_SERVICE == "openai":
            return ChatOpenAI(
                model="gpt-4o",
                temperature=temperature,
                streaming=True,
                callbacks=[StreamingLLMCallbackHandler(websocket)],
                stream_usage=True,
                api_key=config_settings.OPENAI_API_KEY,
            )

        elif config_settings.LLM_SERVICE == "azure-openai":
            return AzureChatOpenAI(
                azure_endpoint=config_settings.AZURE_OPENAI_SETTINGS[model_key]["ENDPOINT"],
                azure_deployment=config_settings.AZURE_OPENAI_SETTINGS[model_key][
                    "DEPLOYMENT"
                ],
                api_key=config_settings.AZURE_OPENAI_SETTINGS[model_key]["API_KEY"],
                api_version=config_settings.AZURE_OPENAI_SETTINGS[model_key]["API_VERSION"],
                temperature=temperature,
                model=config_settings.LLMS.get(model_key, ""),
                streaming=True,
                callbacks=[StreamingLLMCallbackHandler(websocket)],
            )

    except Exception as e:
        logger.error(f"Failed to get chat model with streaming: {e}")
        raise
