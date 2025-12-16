import os
import numpy as np
import base64
import json
import requests

# Try to import the OpenAI Python SDK. If not available, we'll fall back to HTTP requests.
try:
    from openai import OpenAI
    _HAS_OPENAI_SDK = True
except Exception:
    OpenAI = None
    _HAS_OPENAI_SDK = False

# Deepseek configuration: prefer environment variables
# Set DEEPSEEK_API_KEY and optionally DEEPSEEK_API_URL in your environment
DEEPSEEK_API_KEY = "sk-5efe43eb77bd4225bae4d75faa05b09c"
DEEPSEEK_API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')

# instantiate SDK client (base_url should be the Deepseek base) when available
client = None
if _HAS_OPENAI_SDK and DEEPSEEK_API_KEY is not None:
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
    except Exception:
        client = None


def _chat_complete(messages, model: str = 'gpt-4.1'):
    """Helper that chooses the correct model name for Deepseek and calls the SDK.

    If the configured base URL looks like Deepseek, default to the
    Deepseek model name (`deepseek-chat`) unless the caller explicitly
    provided a different model via env `DEEPSEEK_MODEL`.
    """
    effective_model = model
    if 'deepseek' in DEEPSEEK_API_URL.lower():
        effective_model = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')

    resp = _call_chat_completions(effective_model, messages, stream=False)
    return _get_content(resp)


def _normalize_model(model: str) -> str:
    if 'deepseek' in DEEPSEEK_API_URL.lower():
        # Map common caller models to Deepseek's model name
        mapping = {
            'gpt-4.1': os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
            'gpt-4o': os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
            'gpt-4': os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
        }
        return mapping.get(model, os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'))
    return model


def _call_chat_completions(model: str, messages, stream: bool = False, **kwargs):
    """Call chat completions using the OpenAI SDK client if available, otherwise use requests.

    Returns either an SDK response object or a dict similar to the API JSON response.
    """
    effective_model = _normalize_model(model)

    # Prepare messages: some call sites pass `content` as a list of structured
    # items (e.g. {'type':'image_url', ...}). The API expects message.content
    # to be plain text. Serialize structured content into text here.
    def _render_content(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                t = item.get('type')
                if t == 'text':
                    parts.append(item.get('text', ''))
                elif t == 'image_url':
                    # include the image data URL or URL as inline text so the
                    # model receives the image. This was previously passed as
                    # structured content; some backends expect raw text instead.
                    img_obj = item.get('image_url') or item.get('image') or {}
                    url = img_obj.get('url') if isinstance(img_obj, dict) else str(img_obj)
                    parts.append(f"[Image] {url}")
                else:
                    # unknown structured item; stringify
                    parts.append(str(item))
            return "\n".join(parts)
        # fallback
        return str(c)

    prepared_messages = []
    for m in messages:
        role = m.get('role') if isinstance(m, dict) else None
        content = m.get('content') if isinstance(m, dict) else m
        prepared_messages.append({
            'role': role or 'user',
            'content': _render_content(content)
        })

    # Print prepared messages for debugging when calling Deepseek
    # try:
    #     if 'deepseek' in DEEPSEEK_API_URL.lower():
    #         print('\n[Deepseek] Calling chat completions with model:', effective_model)
    #         for i, pm in enumerate(prepared_messages):
    #             c = pm.get('content', '')
    #             # If data URL image large, only show prefix and length
    #             if isinstance(c, str) and 'data:image' in c:
    #                 display = c[:200] + '... (truncated, length=' + str(len(c)) + ')'
    #             else:
    #                 display = c if isinstance(c, str) and len(c) <= 2000 else (str(c)[:2000] + '...')
    #             print(f"  message[{i}] role={pm.get('role')} content={display}")
    #         print('[Deepseek] End messages\n')
    # except Exception:
    #     pass

    # If SDK client available, use it
    if client is not None:
        return client.chat.completions.create(
            model=effective_model,
            messages=prepared_messages,
            stream=stream,
            **{k: v for k, v in kwargs.items() if v is not None}
        )

    # Fallback to HTTP POST
    url = DEEPSEEK_API_URL.rstrip('/') + '/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
    }
    if DEEPSEEK_API_KEY:
        headers['Authorization'] = f'Bearer {DEEPSEEK_API_KEY}'

    payload = {
        'model': effective_model,
        'messages': prepared_messages,
    }
    payload.update({k: v for k, v in kwargs.items() if v is not None})

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise


def _get_content(response):
    """Extract message content from either SDK response object or a dict returned by requests."""
    try:
        # SDK response object
        return response.choices[0].message.content
    except Exception:
        pass
    # dict-like response
    if isinstance(response, dict):
        try:
            return response['choices'][0]['message']['content']
        except Exception:
            pass
    # last resort: try to stringify
    try:
        return str(response)
    except Exception:
        return ''


def get_answer(content):
    return int(content.split('Best Result:')[-1].split('Confidence:')[0].strip(' :*.'))

def get_view(content):
    return int(content.split('Best View:')[-1].strip(' :*.'))

def get_stage(content):
    return int(content.split('Current Stage:')[-1].strip(' :*.'))

def get_grasp(content):
    answer = content.split('Grasp:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_release(content):
    answer = content.split('Release:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_success(content):
    answer = content.split('Satisfied:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_close_gripper(content):
    answer = content.split('Keep Gripper Closed:')[-1].strip(' :*.').lower()
    assert 'yes' == answer or 'no' == answer
    return 'yes' == answer

def get_subgoals(content):
    print(content)

    subgoal_list = []
    goal_id = 0
    content = content.split('Sub Goals:')[-1]
    while True:
        goal_id += 1
        goal = content.split(f'{goal_id}.')[-1].split(f'{goal_id + 1}.')[0].strip(' :*.\n"')
        subgoal_list.append(goal)
        if f'{goal_id + 1}.' not in content:
            break

    return subgoal_list

def get_names(content):
    print(content)

    content = content.split('Objects:')[-1]

    name_list = []
    name_id = 1
    while True:
        if f'{name_id}.' not in content:
            break
        name = content.split(f'{name_id}.')[-1].split(f'{name_id + 1}.')[0].strip(' :*.\n"')
        name_list.append(name)
        name_id += 1

    return name_list

def get_description_list(content, num_results):
    description_list = []
    for idx in range(1, num_results + 1):
        start = content.find(f'Description {idx}:')
        end = content.find(f'Description {idx + 1}:')
        if idx == num_results:
            end = len(content)
        if start == -1 or end == -1:
            print('Description not found')
            return None
        description = content[start:end]
        description_list.append(description)
    return description_list


def get_action(content):
    return content.split('Best Action:')[-1].strip(' :*."')


def simple_generate_response(results: list, system_prompt: str, history = [], grasping = False, model: str = 'gpt-4.1'):
    usr_content = []

    if grasping:
        usr_content.append({"type": "text", "text": f'The gripper is grasping the object now.'})

    for idx, result in enumerate(results):
        usr_content.append({"type": "text", "text": f"This is the obervation of future result {idx + 1}:"})
        image = result[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def simple_select_view(images: list, system_prompt: str, history = None, examples = None, model: str = 'gpt-4.1',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    for idx, image in enumerate(images):
        usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {idx + 1}:"})
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def select_stage(images: list, system_prompt: str, grasping=None, history = None, examples = None, model: str = 'gpt-4.1',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    usr_content.append({"type": "text", "text": f"These are the image obervations of the current state from different views:"})
    if grasping is not None:
        if grasping:
            usr_content.append({"type": "text", "text": f"The gripper is grasping something now."})
        else:
            usr_content.append({"type": "text", "text": f"The gripper is not grasping anything now."})

    for idx, image in enumerate(images):
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def generate_success(images: list, system_prompt: str, grasping=None, history = None, examples = None, model: str = 'gpt-4.1',
                      temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if examples is not None:
        usr_content.append({"type": "text", "text": f"First we will show you some examples."})
        for idx, example in enumerate(examples):
            usr_content.append({"type": "text", "text": f"This is one example:"})
            usr_content.append({"type": "text", "text": f'The goal of robot in this example is: {example[0]}.'})
            for view_id, example_image in enumerate(example[1]):
                usr_content.append({"type": "text", "text": f"This is the image obervation of the current state from view {view_id + 1} in this example:"})
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image}", "detail": "high"}})
            usr_content.append({"type": "text", "text": f'This is the answer of this example: {example[2]}'})

        usr_content.append({"type": "text", "text": f"Below are the real observations you need to handle."})

    usr_content.append({"type": "text", "text": f"These are the image obervations of the current state from different views:"})
    if grasping is not None:
        if grasping:
            usr_content.append({"type": "text", "text": f"The gripper is grasping something now."})
        else:
            usr_content.append({"type": "text", "text": f"The gripper is not grasping anything now."})

    for idx, image in enumerate(images):
        concatenated_image = image[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def generate_subgoals(image, system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    if image is not None:
        if isinstance(image, list):
            usr_content.append({"type": "text", "text": f"These are the image obervations of the initial state:"})
            for img in image:
                usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})
        else:
            usr_content.append({"type": "text", "text": f"This is the image obervation of the initial state:"})
            usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    # print(usr_content)
    # exit(0)
    usr_content.append({"type": "text", "text": f"Please break down the goal into sub-goals for robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def generate_grasp(image, system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f"These are the image obervations after the grasping the object:"})
    for img in image:
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}})
    # usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please tell whether grasping this object align with the goal of the robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def generate_release(image, system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    # careful! not compatible with single image
    usr_content.append({"type": "text", "text": f"These are the image obervations after the releasing the object:"})
    for idx, img in enumerate(image):
        concatenated_image = img[0]
        usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    # image = image[0]
    # usr_content.append({"type": "text", "text": f"This is the image obervation after the releasing the object:"})
    # usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    usr_content.append({"type": "text", "text": f"Please tell whether releasing this object align with the goal of the robot."})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def generate_close_gripper(system_prompt: str, model: str = 'gpt-4.1',
                    temperature: float = 0.7, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []
    usr_content.append({"type": "text", "text": f'Do you think the robot should keep the gripper closed during the whole process?'})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content


def prompt_helper(group_id, queue, prompt, system_prompt, grasping=False):
    try_time = 0
    change = None
    answer = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = simple_generate_response(prompt, system_prompt, grasping=grasping)
            answer = get_answer(content)
            change = True

        except Exception as e:
            print('catched', e)
            pass
    
    if change is None:
        print('Warning: failed to match format')
        answer = 1

    queue.put((group_id, answer, content))


def prompt_release_helper(release_id, queue, prompt, system_prompt):
    try_time = 0
    change = None
    release = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = generate_release(prompt, system_prompt)
            
            release = get_release(content)
            change = True

        except Exception as e:
            print('catched', e)
            pass
    
    if change is None:
        print('Warning: failed to match format')
        release = False

    queue.put((release_id, release, content))


def generate_segment_names(system_prompt: str, image, instruction: str, model: str = 'gpt-4.1',
                      temperature: float = 0.0, max_tokens: int = 16384, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list = None ):
    usr_content = []

    usr_content.append({"type": "text", "text": f"The instruction of the task is: {instruction}"})
    # image = results[0][0]
    usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})

    # for idx, result in enumerate(results[1:]):
    #     usr_content.append({"type": "text", "text": f"This is the image containing multi-view observations of result {idx + 1}."})
    #     if motion_name_list is not None:
    #         usr_content.append({"type": "text", "text": f'The motion direction relative to the robot base is {motion_name_list[idx]}'})
    #     concatenated_image = result[0]
    #     usr_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{concatenated_image}", "detail": "high"}})
    
    payload = {
        "model" : f"{model}",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_content},
        ],
        # "temperature": temperature,
        # "max_tokens": max_tokens,
        # top_p=top_p,
        # frequency_penalty=frequency_penalty,
        # presence_penalty=presence_penalty,
        # stop=stop,
    }

    response = _call_chat_completions(model, payload['messages'], stream=False)
    content = _get_content(response)
    return content
