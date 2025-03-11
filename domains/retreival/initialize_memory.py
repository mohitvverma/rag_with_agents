from langchain.memory import ConversationBufferWindowMemory
from domains.settings import config_settings
from loguru import logger


def initialise_memory_from_chat_context(chat_context, input_key: str = None):
    return __load_chat_context(chat_context, input_key)


def __load_chat_context(chat_context, input_key: str):
    memory = ConversationBufferWindowMemory(
        memory_key=config_settings.CONVERSATIONAL_BUFFER_WINDOW_MEMORY_KEY,
        return_messages=True,
        k=config_settings.LANGCHAIN_MEMORY_BUFFER_WINDOW,
        input_key=input_key or config_settings.CONVERSATIONAL_BUFFER_WINDOW_INPUT_KEY,
    )

    if chat_context is not None:
        logger.info("Loading context from chat context")
        for message in chat_context:
            if message.type == config_settings.CHAT_CONTEXT_HUMAN_MESSAGE_KEY:
                memory.chat_memory.add_user_message(message.content)
            elif message.type == config_settings.CHAT_CONTEXT_AI_MESSAGE_KEY:
                memory.chat_memory.add_ai_message(message.content)

    return memory