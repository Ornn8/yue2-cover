"""Example style listener for an OpenAI-compatible chat endpoint that accepts audio input.

    YUE2_COVER_LISTENER=listener_openai:listen
    YUE2_COVER_LISTENER_URL=https://your-endpoint.example/v1      (the base URL; /chat/completions is appended)
    YUE2_COVER_LISTENER_KEY=...                                    (your own key; read from the environment only)
    YUE2_COVER_LISTENER_MODEL=your-audio-capable-model

This is one way to fill the hook, not a requirement: any function with the signature listen(audio, mime, prompt) -> str
works (see style_words.py). Endpoints differ in how they take audio (input_audio parts, inline data, file uploads); adjust
the payload below for yours. The audio is a ~40 s excerpt; it leaves your machine, so check your provider's terms.
"""
import base64
import json
import os
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None            # never forward the key to another host


def listen(audio, mime, prompt):
    base = os.environ['YUE2_COVER_LISTENER_URL'].rstrip('/')
    payload = {
        'model': os.environ['YUE2_COVER_LISTENER_MODEL'], 'max_tokens': 400, 'temperature': 0.2,
        'messages': [{'role': 'user', 'content': [
            {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(audio).decode('ascii'),
                                                      'format': mime.split('/')[-1]}},
            {'type': 'text', 'text': prompt}]}]}
    request = Request(base + '/chat/completions', data=json.dumps(payload).encode(), method='POST',
                      headers={'Authorization': 'Bearer ' + os.environ['YUE2_COVER_LISTENER_KEY'],
                               'Content-Type': 'application/json'})
    with build_opener(NoRedirect()).open(request, timeout=90) as response:
        body = json.loads(response.read(256 * 1024))
    return body['choices'][0]['message']['content']
