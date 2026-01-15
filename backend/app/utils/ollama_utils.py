import logging
import ollama
from ollama import web_search, web_fetch

async def generate_ollama_stream(messages: list, model: str):
    """Generator function to stream response chunks from Ollama."""
    
    # Streaming was working well for this use case
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True
    )
    for chunk in stream:
        # Yield each content chunk as a string
        yield chunk.message.content


def get_ollama_response_with_web(logger: logging.Logger, messages: list, model: str):
    """Use Ollama to generate responses using the web when appropriate."""
    available_tools = {'web_search': web_search, 'web_fetch': web_fetch}

    while True:
        response = ollama.chat(
            model=model,
            messages=messages,
            think=True,
            tools=[ollama.web_search, ollama.web_fetch]
        )
        if response.message.thinking:
            logger.info('Thinking: ', response.message.thinking)
        if response.message.content:
            logger.info('Content: ', response.message.content)
        messages.append(response.message)
        if response.message.tool_calls:
            logger.info('Tool calls: ', response.message.tool_calls)
            for tool_call in response.message.tool_calls:
                function_to_call = available_tools.get(tool_call.function.name)
            if function_to_call:
                args = tool_call.function.arguments
                result = function_to_call(**args)
                logger.info('Result: ', str(result)[:200]+'...')
                # Result is truncated for limited context lengths
                messages.append({'role': 'tool', 'content': str(result)[:8000 * 4], 'tool_name': tool_call.function.name})
            else:
                messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})
        else:
            break
            
    return messages[-1].content