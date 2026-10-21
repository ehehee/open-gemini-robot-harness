"""VLM client for Gemini 4 Argon, routed through OpenRouter's OpenAI-compatible API.

Gemini 4 Argon was announced 2026-10-13 (model id `gemini-4-argon` on Google's own
platform). We call it here via OpenRouter instead of the Google GenAI SDK directly,
so the model slug is OpenRouter's naming for it rather than Google's. OpenRouter's
exact slug for a same-day model release may differ from the default below -- check
https://openrouter.ai/models once it's listed and override with OPENROUTER_MODEL if
needed.
"""
import base64
import json
import os
from pathlib import Path

from openai import OpenAI

from vlm.action_schema import RobotAction
from vlm.grasp_schema import GraspPoint, HandleGrasp

DEFAULT_MODEL = "google/gemini-4-argon"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are a vision-language controller for a YAM robot arm.
You are given a single camera frame of the arm's workspace and a natural-language
instruction. Decide ONE discrete next action for the arm.

Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:
{
  "reasoning": "<one short sentence>",
  "action": "move_forward" | "move_backward" | "move_left" | "move_right" |
             "move_up" | "move_down" | "open_gripper" | "close_gripper" |
             "home" | "stop",
  "distance_m": <float, meters, only meaningful for move_* actions>,
  "confidence": <float 0-1>
}
"""

GRASP_POINT_SYSTEM_PROMPT = """You are a vision-language controller for a robot
arm, looking at a single top-down overhead camera frame. Given an instruction,
point at the exact pixel where the gripper should grasp the target object.

Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:
{
  "reasoning": "<one short sentence about what you targeted and why>",
  "done": <true if there is nothing left worth grasping, else false>,
  "u": <pixel x of the grasp point>,
  "v": <pixel y of the grasp point>,
  "confidence": <float 0-1>
}
"""

HANDLE_GRASP_SYSTEM_PROMPT = """You are a vision-language grasp planner for a
robot arm with a parallel-jaw gripper, looking at a single top-down overhead
camera frame of a table. Given an instruction naming an object with a handle
(e.g. a brush), find it and plan a top-down grasp ON ITS HANDLE -- not on the
bristles/head. Pick a point near the middle of the handle with clear space on
both sides for the jaws.

Points are [y, x], normalized to 0-1000 over the image (0,0 = top-left).

Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:
{
  "reasoning": "<one short sentence: what you see and where you chose to grasp>",
  "found": <true if the object is visible, else false>,
  "grasp_point": [y, x],
  "handle_axis": [[y, x], [y, x]],
  "confidence": <float 0-1>
}
handle_axis is two points on the handle's centerline, far apart along its
length (e.g. near the handle end and near where it meets the head), so the
gripper can close across the handle.
"""


def _encode_image(image_path: Path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{b64}"


class GeminiArgonClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY in your environment (see .env.example)."
            )
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def infer_action(self, image_path: Path, instruction: str) -> RobotAction:
        image_url = _encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return RobotAction.model_validate(data)

    def infer_handle_grasp(self, image_path: Path, instruction: str) -> HandleGrasp:
        return HandleGrasp.model_validate(self._ask(HANDLE_GRASP_SYSTEM_PROMPT, image_path, instruction))

    def _ask(self, system_prompt: str, image_path: Path, instruction: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                    ],
                },
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)

    def infer_grasp_point(self, image_path: Path, instruction: str) -> GraspPoint:
        image_url = _encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GRASP_POINT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return GraspPoint.model_validate(data)
