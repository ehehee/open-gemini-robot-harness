"""Structured action output the VLM must return for a given scene + instruction."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    HOME = "home"
    STOP = "stop"


class RobotAction(BaseModel):
    reasoning: str = Field(description="Brief explanation of why this action was chosen")
    action: ActionType
    distance_m: Optional[float] = Field(
        default=0.05, description="Cartesian step size in meters for move_* actions"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


ACTION_JSON_SCHEMA = RobotAction.model_json_schema()
