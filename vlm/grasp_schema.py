"""Structured output for the pixel-grounded grasp loop (examples/05).

Unlike RobotAction (discrete move_forward/move_left/...), this asks the VLM to
point at a pixel on the target object directly, matching the
deproject(u, v, z_plane) pattern from Gemini 4 Argon's own robotics writeup --
a precise pixel + known height-plane assumption, rather than the model
reasoning about metric distances itself.
"""
from pydantic import BaseModel, Field


class GraspPoint(BaseModel):
    reasoning: str = Field(description="Brief explanation of what was targeted and why")
    done: bool = Field(description="True if there is nothing left worth grasping in the scene")
    u: float = Field(description="Pixel x of the grasp point")
    v: float = Field(description="Pixel y of the grasp point")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class HandleGrasp(BaseModel):
    """Grasp for an elongated object (brush, tool, utensil) seen top-down: a
    grasp point ON the handle plus two points along the handle's long axis, so
    the jaws can be yawed to close ACROSS it. A point alone (GraspPoint) can't
    say which way a parallel-jaw gripper should face.

    Points are Gemini's native pointing convention, [y, x] normalized to
    0-1000 -- what it's trained to emit, so it's more reliable than asking it
    for raw pixels of an image whose size it only sees after resizing. Convert
    with to_pixels().
    """

    reasoning: str = Field(description="Brief explanation of what was targeted and why")
    found: bool = Field(description="False if the object isn't visible")
    grasp_point: tuple[float, float] = Field(description="[y, x] 0-1000, where to grasp on the handle")
    handle_axis: tuple[tuple[float, float], tuple[float, float]] = Field(
        description="two [y, x] 0-1000 points along the handle's long axis")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @staticmethod
    def _px(pt: tuple[float, float], width: int, height: int) -> tuple[float, float]:
        y, x = pt
        return x / 1000.0 * width, y / 1000.0 * height

    def to_pixels(self, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """(grasp_uv, axis_uv_a, axis_uv_b) in pixel (u, v) of a width x height image."""
        return (self._px(self.grasp_point, width, height),
                self._px(self.handle_axis[0], width, height),
                self._px(self.handle_axis[1], width, height))
