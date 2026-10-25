"""
The robot as rigid bodies in MuJoCo: what actually happens when it drives.

SimRobot still turns DriveBase commands into motion profiles, and from them
the angle each wheel should be at; here, as on a real hub, each motor follows
that angle with a position controller of limited torque. The wheels' grip on
the table then decides what the robot does: it slips, stalls against a block,
or shoves the block out of the way, and its pose comes from the simulation
rather than from the profiles.

World coordinates are SimRobot's, in meters: x right, y forward, z up.
"""

# mujoco star-imports its functions from compiled modules, which pyright
# can't see into.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

WHEEL_RADIUS = 0.028
AXLE_TRACK = 0.112


@dataclass(frozen=True)
class Thing:
    """Something on the table to push around."""

    name: str
    shape: str  # "box" or "sphere"
    size: tuple[float, ...]  # MuJoCo's half sizes, or the radius, in meters
    at: tuple[float, float]  # where it starts on the table
    mass: float  # kg
    color: str


THINGS = (
    Thing("red block", "box", (0.02, 0.02, 0.02), (0.0, 0.35), 0.02, "#d93333"),
    Thing("blue block", "box", (0.02, 0.04, 0.02), (-0.25, 0.3), 0.04, "#3366e6"),
    Thing("ball", "sphere", (0.03,), (0.0, 0.9), 0.05, "#ffcc1a"),
)


def _thing_mjcf(thing: Thing) -> str:
    rgba = " ".join(f"{int(thing.color[i : i + 2], 16) / 255:.3f}" for i in (1, 3, 5))
    z = thing.size[-1] + 0.0005  # resting on the table
    extra = ' condim="6" friction="0.8 0.005 0.002"' if thing.shape == "sphere" else ""
    return (
        f'<body name="{thing.name}" pos="{thing.at[0]} {thing.at[1]} {z}"><freejoint/>'
        f'<geom type="{thing.shape}" size="{" ".join(map(str, thing.size))}" mass="{thing.mass}"'
        f' rgba="{rgba} 1"{extra}/></body>'
    )


# SPIKE Prime's Driving Base 1, in boxes: the motors on their frame, and the
# hub on top of them, both mostly behind the axle, which is the body's origin,
# so that the weight rests between the wheels and the caster ball at the back.
# Sizes and places follow models/driving_base.ldr.
MJCF = f"""
<mujoco model="pybricks driving base">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <default>
    <geom friction="0.8 0.005 0.0001"/>
    <default class="tire">
      <!-- An ellipsoid touches the table at one point, as a rounded tire does; a
           cylinder touches at both rim edges and scrubs when the robot turns. -->
      <geom type="ellipsoid" size="0.007 {WHEEL_RADIUS} {WHEEL_RADIUS}" friction="1.2 0.002 0.0001"
            condim="6" mass="0.03"/>
    </default>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="10 10 0.1"/>
    <body name="robot" pos="0 0 0.0005">
      <freejoint name="robot"/>
      <geom name="motors" type="box" size="0.028 0.04 0.016" pos="0 -0.028 0.024" mass="0.25"/>
      <geom name="hub" type="box" size="0.028 0.044 0.016" pos="0 -0.032 0.056" mass="0.2"/>
      <!-- Frictionless: a contact takes the larger condim of its two geoms, so
           without priority the table's would make the caster drag. -->
      <geom name="caster" type="sphere" size="0.0095" pos="0 -0.072 0.0095" condim="1" priority="1" mass="0.01"/>
      <body name="left wheel" pos="-0.056 0 {WHEEL_RADIUS}">
        <joint name="left" type="hinge" axis="1 0 0" damping="0.002" armature="0.0004"/>
        <geom class="tire"/>
      </body>
      <body name="right wheel" pos="0.056 0 {WHEEL_RADIUS}">
        <joint name="right" type="hinge" axis="1 0 0" damping="0.002" armature="0.0004"/>
        <geom class="tire"/>
      </body>
    </body>
    {"".join(_thing_mjcf(thing) for thing in THINGS)}
  </worldbody>
  <actuator>
    <!-- Like a SPIKE motor holding a reference angle: stiff, with a torque limit. -->
    <position name="left" joint="left" kp="1.5" kv="0.03" forcerange="-0.18 0.18"/>
    <position name="right" joint="right" kp="1.5" kv="0.03" forcerange="-0.18 0.18"/>
  </actuator>
</mujoco>
"""


@dataclass
class Pose:
    position: np.ndarray  # meters
    rotation: np.ndarray  # 3x3, body to world


@dataclass
class World:
    """Everything the 3D view draws, at one moment."""

    robot: Pose
    heading: float  # degrees clockwise from +y
    wheels: tuple[float, float]  # degrees, forward positive
    things: dict[str, Pose]


class Physics:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.robot = self.model.body("robot").id
        self.wheels = [self.model.joint(name).qposadr[0] for name in ("left", "right")]
        self.things = {thing.name: self.model.body(thing.name).id for thing in THINGS}
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, dt: float, left_deg: float, right_deg: float) -> None:
        """
        Advances by dt seconds with each motor holding its reference angle.
        A wheel turning forward is a negative rotation about +x.
        """
        self.data.ctrl[:] = (-math.radians(left_deg), -math.radians(right_deg))
        for _ in range(max(1, round(dt / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def pose(self, body: int) -> Pose:
        return Pose(self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy())

    def robot_pose(self) -> Pose:
        return self.pose(self.robot)

    def wheel_angles(self) -> tuple[float, float]:
        """Each wheel's rotation, in degrees, forward positive."""
        return tuple(-math.degrees(self.data.qpos[i]) for i in self.wheels)  # type: ignore[return-value]

    def heading(self) -> float:
        """Degrees clockwise from +y, as SimRobot measures it."""
        forward = self.data.xmat[self.robot].reshape(3, 3)[:, 1]
        return math.degrees(math.atan2(forward[0], forward[1]))

    def world(self) -> World:
        return World(
            self.robot_pose(),
            self.heading(),
            self.wheel_angles(),
            {name: self.pose(body) for name, body in self.things.items()},
        )
