"""Robot commands in the shape of Pybricks' DriveBase, and a simulated robot.

The commands mirror pybricks.robotics.DriveBase: straight(mm), turn(deg),
drive(speed, turn_rate) for a time, and stop(). A real hub can run them as
they are later (M2); SimRobot runs them here with differential-drive
kinematics and the same kind of speed and acceleration limits.

Coordinates: millimeters, x to the right and y forward from the start; the
heading is in degrees, 0 = facing +y, positive = clockwise, as Pybricks'
turn() is.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Straight:
    distance: float  # mm, negative backwards

    def __str__(self) -> str:
        return f"straight({self.distance:g} mm)"


@dataclass(frozen=True)
class Turn:
    angle: float  # degrees, positive clockwise (right)

    def __str__(self) -> str:
        return f"turn({self.angle:g}°)"


@dataclass(frozen=True)
class Curve:
    """An arc: like Pybricks' DriveBase.curve(radius, angle).

    A negative radius drives backwards, as in Pybricks; the angle's sign is the
    direction the heading turns, positive right (clockwise), as turn() does.
    (Check Pybricks' own sign convention for curve() when this reaches a real hub.)
    """

    radius: float  # mm, negative backwards
    angle: float  # degrees along the circle, positive clockwise (right)

    def __str__(self) -> str:
        return f"curve(r {self.radius:g} mm, {self.angle:g}°)"


@dataclass(frozen=True)
class Home:
    """Back to where the robot started, facing the way it started."""

    def __str__(self) -> str:
        return "home()"


@dataclass(frozen=True)
class Drive:
    speed: float  # mm/s
    turn_rate: float  # deg/s, positive clockwise
    duration: float  # s

    def __str__(self) -> str:
        return f"drive({self.speed:g} mm/s, {self.turn_rate:g}°/s, {self.duration:g} s)"


@dataclass(frozen=True)
class Stop:
    def __str__(self) -> str:
        return "stop()"


@dataclass(frozen=True)
class SetSpeed:
    """Scales the straight and turn speeds, for "faster" and "slower"."""

    factor: float

    def __str__(self) -> str:
        return f"speed ×{self.factor:g}"


Command = Straight | Turn | Curve | Drive | Stop | SetSpeed | Home


@dataclass
class Geometry:
    """SPIKE Prime Driving Base 2 by default."""

    wheel_diameter: float = 56.0  # mm
    axle_track: float = 112.0  # mm


@dataclass
class Limits:
    """Like Pybricks' DriveBase.settings() defaults for a robot this size."""

    straight_speed: float = 300.0  # mm/s
    straight_acceleration: float = 600.0  # mm/s²
    turn_rate: float = 180.0  # deg/s
    turn_acceleration: float = 720.0  # deg/s²


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0  # mm/s, current
    turn_rate: float = 0.0  # deg/s, current
    left_angle: float = 0.0  # wheel rotation, degrees
    right_angle: float = 0.0
    distance: float = 0.0  # odometer, mm
    current: Command | None = None
    queued: list[Command] = field(default_factory=list)


def _approach(value: float, target: float, step: float) -> float:
    if value < target:
        return min(value + step, target)
    return max(value - step, target)


class SimRobot:
    """Runs commands one after another; call step(dt) from a timer."""

    def __init__(self, geometry: Geometry | None = None, limits: Limits | None = None) -> None:
        self.geometry = geometry or Geometry()
        self.limits = limits or Limits()
        self.speed_factor = 1.0
        self._state = State()
        self._queue: deque[Command] = deque()
        self._current: Command | None = None
        self._remaining = 0.0  # mm, degrees or seconds left of the current command
        self._lock = threading.Lock()

    # ----- commands --------------------------------------------------------

    def run(self, commands: list[Command]) -> None:
        """Queues commands; a Stop among them clears everything before it."""
        with self._lock:
            for command in commands:
                if isinstance(command, Stop):
                    self._queue.clear()
                    self._current = command
                    self._remaining = 0.0
                elif isinstance(command, SetSpeed):
                    self.speed_factor = min(3.0, max(0.25, self.speed_factor * command.factor))
                else:
                    self._queue.append(command)

    def stop(self) -> None:
        self.run([Stop()])

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._current = None
            self._state = State()
            self.speed_factor = 1.0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None or bool(self._queue)

    def state(self) -> State:
        with self._lock:
            s = self._state
            return State(
                s.x,
                s.y,
                s.heading,
                s.speed,
                s.turn_rate,
                s.left_angle,
                s.right_angle,
                s.distance,
                self._current,
                list(self._queue),
            )

    # ----- simulation ------------------------------------------------------

    def step(self, dt: float) -> None:
        with self._lock:
            self._step(dt)

    def _step(self, dt: float) -> None:
        s, lim = self._state, self.limits
        vmax = lim.straight_speed * self.speed_factor
        wmax = lim.turn_rate * self.speed_factor

        if self._current is None and self._queue and isinstance(self._queue[0], Home):
            # Where home is depends on where the robot is by then, so it is
            # worked out only now: face the start, drive there, face forward.
            self._queue.popleft()
            distance = math.hypot(s.x, s.y)
            steps: list[Command] = []
            if distance > 1:
                bearing = math.degrees(math.atan2(-s.x, -s.y))
                steps += [Turn((bearing - s.heading + 180) % 360 - 180), Straight(distance)]
                final = (0 - bearing + 180) % 360 - 180
            else:
                final = (0 - s.heading + 180) % 360 - 180
            if abs(final) > 0.01:
                steps.append(Turn(final))
            self._queue.extendleft(reversed([c for c in steps if not (isinstance(c, Turn) and abs(c.angle) < 0.01)]))
        if self._current is None and self._queue:
            self._current = self._queue.popleft()
            c = self._current
            self._remaining = (
                abs(c.distance)
                if isinstance(c, Straight)
                else (abs(c.angle) if isinstance(c, (Turn, Curve)) else c.duration if isinstance(c, Drive) else 0.0)
            )

        c = self._current
        speed_target = turn_target = 0.0
        if isinstance(c, Curve):
            # Steer by the angle left, like a turn, but no faster than the wheels
            # can carry the body around the circle; the speed follows the turn rate.
            radians_per_degree = math.pi / 180
            radius = abs(c.radius)
            w_cap = min(wmax, vmax / (radius * radians_per_degree))
            reachable = math.sqrt(2 * lim.turn_acceleration * self._remaining)
            s.turn_rate = _approach(
                s.turn_rate, math.copysign(min(w_cap, reachable), c.angle), lim.turn_acceleration * dt
            )
            rotate = math.copysign(min(abs(s.turn_rate * dt), self._remaining), c.angle)
            s.speed = math.copysign(abs(s.turn_rate) * radians_per_degree * radius, c.radius)
            self._remaining -= abs(rotate)
            self._advance(math.copysign(abs(rotate) * radians_per_degree * radius, c.radius), rotate)
            if self._remaining <= 1e-6:
                s.speed = s.turn_rate = 0.0
                self._current = None
            return
        if isinstance(c, Straight):
            # Trapezoid: the fastest speed that can still stop in what is left.
            reachable = math.sqrt(2 * lim.straight_acceleration * self._remaining)
            speed_target = math.copysign(min(vmax, reachable), c.distance)
        elif isinstance(c, Turn):
            reachable = math.sqrt(2 * lim.turn_acceleration * self._remaining)
            turn_target = math.copysign(min(wmax, reachable), c.angle)
        elif isinstance(c, Drive):
            speed_target = max(-vmax, min(vmax, c.speed * self.speed_factor))
            turn_target = max(-wmax, min(wmax, c.turn_rate * self.speed_factor))

        s.speed = _approach(s.speed, speed_target, lim.straight_acceleration * dt)
        s.turn_rate = _approach(s.turn_rate, turn_target, lim.turn_acceleration * dt)

        move = s.speed * dt
        rotate = s.turn_rate * dt
        if isinstance(c, Straight):
            move = math.copysign(min(abs(move), self._remaining), c.distance)
            self._remaining -= abs(move)
        elif isinstance(c, Turn):
            rotate = math.copysign(min(abs(rotate), self._remaining), c.angle)
            self._remaining -= abs(rotate)
        elif isinstance(c, Drive):
            self._remaining -= dt

        self._advance(move, rotate)

        done = (
            (isinstance(c, (Straight, Turn)) and self._remaining <= 1e-6)
            or (isinstance(c, Drive) and self._remaining <= 0)
            or (isinstance(c, Stop) and s.speed == 0 and s.turn_rate == 0)
        )
        if done:
            if isinstance(c, (Straight, Turn)):
                # Settle exactly, as the real DriveBase does when it holds.
                s.speed = s.turn_rate = 0.0
            self._current = None

    def _advance(self, move: float, rotate: float) -> None:
        """Moves the body move mm along its heading while turning rotate degrees."""
        s = self._state
        # Along the heading halfway through the rotation (midpoint rule).
        mid = math.radians(s.heading + rotate / 2)
        s.x += move * math.sin(mid)
        s.y += move * math.cos(mid)
        s.heading = (s.heading + rotate + 180) % 360 - 180
        s.distance += abs(move)

        # Each wheel's arc: clockwise turning drives the left wheel forward.
        arc = math.radians(rotate) * self.geometry.axle_track / 2
        to_degrees = 360 / (math.pi * self.geometry.wheel_diameter)
        s.left_angle += (move + arc) * to_degrees
        s.right_angle += (move - arc) * to_degrees
