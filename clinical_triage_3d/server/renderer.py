"""
3D Ward Renderer — Pygame + OpenGL headless rendering.
Produces 84x84 (training) or 256x256 (eval) JPEG frames.
No Unreal Engine or Unity required — pure Python, runs in Docker.

Architecture follows CARLA: camera captures scene -> base64 JPEG -> observation.
"""
from __future__ import annotations
import base64
import io
import math
import os
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")   # headless — no display needed
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Ward layout constants (metres)
WARD_WIDTH = 12.0
WARD_DEPTH = 8.0
WARD_HEIGHT = 3.0

BED_POSITIONS = {
    "bed_1": (2.0, 0.0, 2.0),
    "bed_2": (2.0, 0.0, 6.0),
    "bed_3": (10.0, 0.0, 2.0),
    "bed_4": (10.0, 0.0, 6.0),
}

STATION_POSITION = (6.0, 0.0, 4.0)
CART_POSITION = (6.0, 0.0, 1.0)

ALERT_COLORS = {
    "critical": (1.0, 0.1, 0.1),
    "warning": (1.0, 0.8, 0.0),
    "stable": (0.1, 0.8, 0.2),
    "unknown": (0.6, 0.6, 0.6),
    "deceased": (0.2, 0.2, 0.2),
}

POSITION_CAMERA_MAP = {
    "bed_1":          {"eye": (2.0, 1.6, 0.5),  "look": (2.0, 0.8, 2.0)},
    "bed_2":          {"eye": (2.0, 1.6, 7.5),  "look": (2.0, 0.8, 6.0)},
    "bed_3":          {"eye": (10.0, 1.6, 0.5), "look": (10.0, 0.8, 2.0)},
    "bed_4":          {"eye": (10.0, 1.6, 7.5), "look": (10.0, 0.8, 6.0)},
    "nurses_station": {"eye": (6.0, 1.8, 4.0),  "look": (6.0, 0.8, 4.0)},
    "equipment_cart": {"eye": (6.0, 1.8, 1.0),  "look": (6.0, 0.8, 2.0)},
    "exit":           {"eye": (6.0, 1.8, 8.0),  "look": (6.0, 0.8, 4.0)},
    "entry":          {"eye": (6.0, 1.8, -0.5), "look": (6.0, 0.8, 4.0)},
}


class WardRenderer:
    """
    Renders the 3D emergency department ward using Pygame + OpenGL.
    Produces base64 JPEG frames for the RL agent's visual observation.
    """

    def __init__(self, width: int = 84, height: int = 84):
        self.width = width
        self.height = height
        self._initialized = False

    def _init(self):
        if self._initialized:
            return
        pygame.init()
        pygame.display.set_mode(
            (self.width, self.height),
            DOUBLEBUF | OPENGL | pygame.NOFRAME,
        )
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 3.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        self._initialized = True

    def render_frame(
        self,
        agent_position: str,
        patient_states: Dict[str, Dict],
    ) -> str:
        """
        Render one frame from the agent's current position.
        Returns base64-encoded JPEG string.
        """
        self._init()

        glClearColor(0.85, 0.92, 0.95, 1.0)   # hospital ceiling colour
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set camera based on agent position
        cam = POSITION_CAMERA_MAP.get(
            agent_position, POSITION_CAMERA_MAP["nurses_station"]
        )
        gluLookAt(
            *cam["eye"],
            *cam["look"],
            0.0, 1.0, 0.0,
        )

        self._draw_floor()
        self._draw_walls()
        self._draw_ceiling_lights()
        self._draw_nurses_station()
        self._draw_equipment_cart()

        for bed_id, pos in BED_POSITIONS.items():
            patient = patient_states.get(bed_id, {})
            alert = patient.get("alert_level", "unknown")
            self._draw_patient_bed(pos, alert, bed_id)

        pygame.display.flip()

        # Capture frame to base64 JPEG
        data = pygame.image.tostring(pygame.display.get_surface(), "RGB")
        img = pygame.image.fromstring(data, (self.width, self.height), "RGB")
        buf = io.BytesIO()
        pygame.image.save(img, buf, "JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _draw_floor(self):
        glDisable(GL_LIGHTING)
        glBegin(GL_QUADS)
        # Tile pattern — alternating light/dark
        for i in range(int(WARD_WIDTH)):
            for j in range(int(WARD_DEPTH)):
                c = 0.92 if (i + j) % 2 == 0 else 0.82
                glColor3f(c, c, c)
                glVertex3f(i, 0.0, j)
                glVertex3f(i + 1, 0.0, j)
                glVertex3f(i + 1, 0.0, j + 1)
                glVertex3f(i, 0.0, j + 1)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_walls(self):
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.94, 0.94, 0.92, 1.0])
        glBegin(GL_QUADS)
        # Back wall
        glVertex3f(0, 0, WARD_DEPTH)
        glVertex3f(WARD_WIDTH, 0, WARD_DEPTH)
        glVertex3f(WARD_WIDTH, WARD_HEIGHT, WARD_DEPTH)
        glVertex3f(0, WARD_HEIGHT, WARD_DEPTH)
        # Left wall
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, WARD_DEPTH)
        glVertex3f(0, WARD_HEIGHT, WARD_DEPTH)
        glVertex3f(0, WARD_HEIGHT, 0)
        # Right wall
        glVertex3f(WARD_WIDTH, 0, 0)
        glVertex3f(WARD_WIDTH, 0, WARD_DEPTH)
        glVertex3f(WARD_WIDTH, WARD_HEIGHT, WARD_DEPTH)
        glVertex3f(WARD_WIDTH, WARD_HEIGHT, 0)
        glEnd()

    def _draw_ceiling_lights(self):
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.9)
        for x in [3.0, 6.0, 9.0]:
            for z in [2.0, 6.0]:
                glBegin(GL_QUADS)
                glVertex3f(x - 0.3, WARD_HEIGHT - 0.02, z - 0.6)
                glVertex3f(x + 0.3, WARD_HEIGHT - 0.02, z - 0.6)
                glVertex3f(x + 0.3, WARD_HEIGHT - 0.02, z + 0.6)
                glVertex3f(x - 0.3, WARD_HEIGHT - 0.02, z + 0.6)
                glEnd()
        glEnable(GL_LIGHTING)

    def _draw_patient_bed(
        self, pos: Tuple, alert: str, bed_id: str
    ):
        x, y, z = pos
        colour = ALERT_COLORS.get(alert, ALERT_COLORS["unknown"])

        # Bed frame (grey metal)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.7, 0.7, 0.72, 1.0])
        self._draw_box(x - 0.4, 0.0, z - 0.9, 0.8, 0.55, 1.8)

        # Mattress (white)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.96, 0.96, 0.96, 1.0])
        self._draw_box(x - 0.38, 0.55, z - 0.88, 0.76, 0.1, 1.76)

        # Patient body (alert colour = clinical state)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [*colour, 1.0])
        self._draw_box(x - 0.2, 0.65, z - 0.7, 0.4, 0.2, 1.4)

        # Head (sphere approximated as box)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.88, 0.72, 0.60, 1.0])
        self._draw_box(x - 0.12, 0.65, z - 0.82, 0.24, 0.24, 0.24)

        # Monitor stand (only for critical/warning)
        if alert in ("critical", "warning"):
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.0])
            self._draw_box(x + 0.5, 0.0, z, 0.05, 1.5, 0.05)
            # Monitor screen (glowing colour)
            glDisable(GL_LIGHTING)
            glColor3f(*colour)
            self._draw_box(x + 0.42, 1.35, z - 0.2, 0.15, 0.3, 0.4)
            glEnable(GL_LIGHTING)

        # IV drip for immediate patients
        if alert == "critical":
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.9, 0.9, 1.0, 1.0])
            self._draw_box(x - 0.55, 0.0, z + 0.5, 0.03, 1.6, 0.03)
            self._draw_box(x - 0.65, 1.5, z + 0.4, 0.12, 0.2, 0.2)

    def _draw_nurses_station(self):
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.75, 0.82, 0.85, 1.0])
        self._draw_box(5.0, 0.0, 3.5, 2.0, 1.0, 1.0)
        # Computer screen
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.7, 1.0)
        self._draw_box(5.8, 1.0, 3.6, 0.5, 0.35, 0.04)
        glEnable(GL_LIGHTING)

    def _draw_equipment_cart(self):
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.85, 0.85, 0.7, 1.0])
        self._draw_box(5.5, 0.0, 0.8, 1.0, 0.9, 0.5)
        # Drawers
        for dy in [0.15, 0.45, 0.72]:
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
            self._draw_box(5.52, dy, 0.82, 0.96, 0.22, 0.04)

    @staticmethod
    def _draw_box(x, y, z, w, h, d):
        """Draw a solid box at (x,y,z) with dimensions (w,h,d)."""
        glBegin(GL_QUADS)
        # Front face
        glNormal3f(0, 0, -1)
        glVertex3f(x, y, z); glVertex3f(x+w, y, z)
        glVertex3f(x+w, y+h, z); glVertex3f(x, y+h, z)
        # Back face
        glNormal3f(0, 0, 1)
        glVertex3f(x, y, z+d); glVertex3f(x+w, y, z+d)
        glVertex3f(x+w, y+h, z+d); glVertex3f(x, y+h, z+d)
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(x, y, z); glVertex3f(x, y, z+d)
        glVertex3f(x, y+h, z+d); glVertex3f(x, y+h, z)
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(x+w, y, z); glVertex3f(x+w, y, z+d)
        glVertex3f(x+w, y+h, z+d); glVertex3f(x+w, y+h, z)
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(x, y+h, z); glVertex3f(x+w, y+h, z)
        glVertex3f(x+w, y+h, z+d); glVertex3f(x, y+h, z+d)
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(x, y, z); glVertex3f(x+w, y, z)
        glVertex3f(x+w, y, z+d); glVertex3f(x, y, z+d)
        glEnd()

    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False
