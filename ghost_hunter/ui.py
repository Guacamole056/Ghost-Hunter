"""Pygame desktop UI: human play, AI demonstration, belief heatmap, and scores."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
try:
    import pygame
except ImportError as exc:
    raise ImportError("Pygame is required for the game. Run: pip install -r requirements.txt") from exc

from .agents import HunterAgent
from .core import GameSession, cell_index, position_for
from .evaluation import PROJECT_ROOT


class GameUI:
    """Compact, keyboard-friendly desktop UI with no external image assets."""

    WIDTH, HEIGHT = 1250, 840
    BOARD_PANEL_X, BOARD_PANEL_Y, BOARD_PANEL_SIZE = 51, 171, 581
    BOARD_X, BOARD_Y, BOARD_SIZE = 88, 204, 532
    PANEL_X, PANEL_W = 668, 517
    ACCURACY_PRESETS = (0.40, 0.60, 0.80, 1.00)
    BG = (13, 23, 30)
    SURFACE = (20, 35, 44)
    SURFACE2 = (27, 46, 55)
    OUTLINE = (44, 68, 78)
    TEXT = (230, 239, 237)
    MUTED = (143, 168, 173)
    MINT = (94, 216, 189)
    LIGHT = (172, 244, 222)
    DANGER = (233, 145, 122)
    BOARD_BG = (24, 42, 50)
    def __init__(self, *, mode: str = "human", grid_size: int = 10,
                 max_turns: int = 80, sensor_accuracy: float = 0.60,
                 seed: int = 42):
        if not 2 <= grid_size <= 18:
            raise ValueError("For readability, GUI grid size must be 2–18.")
        pygame.init()
        pygame.display.set_caption("Ghost Hunter  |  Probabilistic Tracking Lab")
        info = pygame.display.Info()
        # Size the window to smaller laptop screens while preserving the 1250x840
        # logical canvas. Resizing uses letterboxing rather than distortion.
        avail_w = max(780, (info.current_w or self.WIDTH + 80) - 80)
        avail_h = max(560, (info.current_h or self.HEIGHT + 100) - 100)
        initial_scale = min(1.0, avail_w / self.WIDTH, avail_h / self.HEIGHT)
        window_size = (int(self.WIDTH * initial_scale), int(self.HEIGHT * initial_scale))
        self.window = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = {
            "display": pygame.font.SysFont("segoeui", 39, bold=True),
            "title": pygame.font.SysFont("segoeui", 25, bold=True),
            "section": pygame.font.SysFont("segoeui", 18, bold=True),
            "body": pygame.font.SysFont("segoeui", 17),
            "small": pygame.font.SysFont("segoeui", 14),
            "mini": pygame.font.SysFont("segoeui", 12),
            "mono": pygame.font.SysFont("consolas", 18, bold=True),
        }
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.sensor_accuracy = sensor_accuracy
        self.seed = seed
        self.heatmap_enabled = True
        self._used_heatmap_in_episode = True
        self.show_scores = False
        self.autoplay = False
        self.auto_interval = 300
        self.last_ai_tick = 0
        self.running = True
        self.status = "The ghost is hidden. Use the noisy reading to track it."
        self.hitboxes: dict[str, pygame.Rect] = {}
        self._modal_rect: pygame.Rect | None = None
        self._result_saved = False
        self.agent: HunterAgent | None = HunterAgent(mode) if mode != "human" else None
        self.session = GameSession(mode=mode, grid_size=grid_size,
                                   max_turns=max_turns,
                                   sensor_accuracy=sensor_accuracy, seed=seed)

    def _viewport(self) -> pygame.Rect:
        ww, wh = self.window.get_size()
        scale = min(ww / self.WIDTH, wh / self.HEIGHT)
        vw, vh = max(1, round(self.WIDTH * scale)), max(1, round(self.HEIGHT * scale))
        return pygame.Rect((ww - vw) // 2, (wh - vh) // 2, vw, vh)

    def _to_logical(self, pos: tuple[int, int]) -> tuple[int, int]:
        view = self._viewport()
        if not view.collidepoint(pos):
            return (-1, -1)
        return (round((pos[0] - view.x) * self.WIDTH / view.width),
                round((pos[1] - view.y) * self.HEIGHT / view.height))

    def _text(self, value: str, x: int, y: int, *, font: str = "body",
              color=None, align: str = "left") -> pygame.Rect:
        surface = self.fonts[font].render(str(value), True, color or self.TEXT)
        rect = surface.get_rect()
        if align == "center":
            rect.center = (x, y)
        elif align == "right":
            rect.topright = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surface, rect)
        return rect

    def _panel(self, x: int, y: int, w: int, h: int, color=None, radius=15,
               outline=False, outline_color=None) -> None:
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, color or self.SURFACE, rect, border_radius=radius)
        if outline:
            pygame.draw.rect(self.screen, outline_color or self.OUTLINE, rect,
                             width=1, border_radius=radius)

    def _button(self, key: str, label: str, rect: tuple[int, int, int, int], *,
                active: bool = False, strong: bool = False,
                modal: bool = False) -> None:
        area = pygame.Rect(*rect)
        interactive = modal or not self.show_scores
        if interactive:
            self.hitboxes[key] = area
        hovered = interactive and area.collidepoint(self._to_logical(pygame.mouse.get_pos()))
        if active or strong:
            fill = self.MINT if active else (55, 164, 147)
            fg = self.BG
        else:
            fill = (42, 66, 75) if hovered else self.SURFACE2
            fg = self.TEXT
        pygame.draw.rect(self.screen, fill, area, border_radius=9)
        if not active and not strong:
            pygame.draw.rect(self.screen, self.OUTLINE, area, 1, border_radius=9)
        self._text(label, area.centerx, area.centery, font="small", color=fg, align="center")

    def new_game(self, mode: str | None = None, *, next_seed: bool = False) -> None:
        if next_seed:
            self.seed += 1
        mode = mode or self.session.mode
        self.session = GameSession(mode=mode, grid_size=self.grid_size,
                                   max_turns=self.max_turns,
                                   sensor_accuracy=self.sensor_accuracy,
                                   seed=self.seed)
        self.agent = HunterAgent(mode) if mode != "human" else None
        self.autoplay = False
        self._result_saved = False
        self._used_heatmap_in_episode = self.heatmap_enabled
        self.status = f"New {self._mode_label()} episode (seed {self.seed}, {self.sensor_accuracy * 100:.0f}% accuracy)."

    def _set_accuracy(self, accuracy: float) -> None:
        if abs(self.sensor_accuracy - accuracy) < 1e-6:
            return
        self.sensor_accuracy = accuracy
        self.new_game()
        self.status = f"Sensor accuracy set to {accuracy * 100:.0f}% exact. Started fresh episode."

    def _adjust_speed(self, delta_ms: int) -> None:
        self.auto_interval = max(50, min(1000, self.auto_interval + delta_ms))
        self.status = f"Autoplay speed set to {self.auto_interval} ms / turn."

    def _toggle_heatmap(self) -> None:
        self.heatmap_enabled = not self.heatmap_enabled
        if self.heatmap_enabled and not self.session.finished:
            self._used_heatmap_in_episode = True

    def _mode_label(self) -> str:
        return {"human": "Human", "ai": "HMM AI", "baseline": "Sensor-only"}[self.session.mode]

    def _human_score_path(self) -> Path:
        return PROJECT_ROOT / "results" / "human_scores.json"

    def _save_human_score(self) -> None:
        if self._result_saved or not self.session.finished or self.session.mode != "human":
            return
        path = self._human_score_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            data.append({"date": datetime.now().isoformat(timespec="seconds"),
                         "seed": self.seed,
                         "sensor_accuracy": self.sensor_accuracy,
                         "heatmap_used": self._used_heatmap_in_episode,
                         **self.session.result()})
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError:
            self.status = "Score not saved (folder is read-only)."
        else:
            self._result_saved = True

    def take_turn(self, action: str) -> None:
        if self.session.finished:
            return
        if not self.session.take_action(action):
            self.status = "Invalid move: board edge reached."
            return
        if self.session.captured:
            gr, gc = self.session.ghost_position
            self.status = f"Captured at R{gr + 1:02} C{gc + 1:02} in {self.session.turns} turns!"
        elif self.session.finished:
            gr, gc = self.session.ghost_position
            self.status = f"Turn limit reached. Ghost revealed at R{gr + 1:02} C{gc + 1:02}."
        elif self.session.last_capture_missed:
            self.status = "Capture missed. Belief updated; ghost moved again."
        else:
            self.status = f"Turn {self.session.turns}: the sensor has a new reading."
        if self.session.finished:
            self.autoplay = False
            self._save_human_score()

    def _ai_turn(self) -> None:
        if self.session.mode == "human" or self.session.finished:
            return
        if self.agent is None or self.agent.strategy != self.session.mode:
            self.agent = HunterAgent(self.session.mode)
        action = self.agent.choose_action(self.session)
        self.take_turn(action)

    def _screenshot(self) -> None:
        folder = PROJECT_ROOT / "results" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"ghost_hunter_{self.session.mode}_seed{self.seed}_turn{self.session.turns}.png"
        pygame.image.save(self.screen, str(path))
        self.status = f"Screenshot saved: {path.name}"

    def _click(self, key: str) -> None:
        if key == "close_scores":
            self.show_scores = False
        elif key in ("human", "ai", "baseline"):
            self.new_game(key)
        elif key == "heatmap":
            self._toggle_heatmap()
        elif key == "restart":
            self.new_game()
        elif key == "new_seed":
            self.new_game(next_seed=True)
        elif key == "scores":
            self.show_scores = not self.show_scores
        elif key == "step":
            self._ai_turn()
        elif key == "capture":
            if self.session.mode == "human":
                self.take_turn("capture")
        elif key == "speed_slower":
            self._adjust_speed(50)
        elif key == "speed_faster":
            self._adjust_speed(-50)
        elif key == "autoplay":
            if self.session.mode != "human" and not self.session.finished:
                self.autoplay = not self.autoplay
                self.last_ai_tick = pygame.time.get_ticks()
        elif key.startswith("acc_"):
            self._set_accuracy(float(key.split("_", 1)[1]))

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.show_scores:
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_TAB, pygame.K_ESCAPE):
                    self.show_scores = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = self._to_logical(event.pos)
                    close_rect = self.hitboxes.get("close_scores")
                    if close_rect and close_rect.collidepoint(pos):
                        self.show_scores = False
                    elif self._modal_rect and not self._modal_rect.collidepoint(pos):
                        self.show_scores = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = self._to_logical(event.pos)
                for key, rect in tuple(self.hitboxes.items()):
                    if rect.collidepoint(pos):
                        self._click(key)
                        break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.new_game({pygame.K_1: "human", pygame.K_2: "ai",
                                   pygame.K_3: "baseline"}[event.key])
                elif event.key == pygame.K_r:
                    self.new_game()
                elif event.key == pygame.K_n:
                    self.new_game(next_seed=True)
                elif event.key == pygame.K_h:
                    self._toggle_heatmap()
                elif event.key == pygame.K_p:
                    self._screenshot()
                elif event.key == pygame.K_TAB:
                    self.show_scores = True
                elif event.key == pygame.K_SPACE:
                    self._ai_turn()
                elif event.key == pygame.K_a and self.session.mode != "human":
                    if not self.session.finished:
                        self.autoplay = not self.autoplay
                        self.last_ai_tick = pygame.time.get_ticks()
                elif event.key == pygame.K_LEFTBRACKET:
                    self._adjust_speed(50)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self._adjust_speed(-50)
                elif self.session.mode == "human":
                    actions = {
                        pygame.K_UP: "up", pygame.K_w: "up",
                        pygame.K_DOWN: "down", pygame.K_s: "down",
                        pygame.K_LEFT: "left", pygame.K_a: "left",
                        pygame.K_RIGHT: "right", pygame.K_d: "right",
                        pygame.K_c: "capture",
                    }
                    action = actions.get(event.key)
                    if action:
                        self.take_turn(action)

    def _draw_header(self) -> None:
        ghost_rect = self._text("GHOST", 52, 30, font="display", color=self.TEXT)
        self._text("/ HUNTER", ghost_rect.right + 10, 30, font="display", color=self.MINT)
        self._text("PROBABILISTIC TRACKING LAB", 54, 84, font="small", color=self.MUTED)
        self._text("HIDDEN MARKOV MODEL  /  FORWARD INFERENCE", 1185, 42,
                   font="small", color=self.MUTED, align="right")
        self._button("scores", "TAB   SCOREBOARD", (929, 76, 256, 30))
        pygame.draw.line(self.screen, self.OUTLINE, (51, 116), (1185, 116), 1)
        if self.session.finished:
            accent = self.MINT if self.session.captured else self.DANGER
            bg = (19, 44, 46) if self.session.captured else (48, 28, 30)
            self._panel(self.BOARD_PANEL_X, 127, self.BOARD_PANEL_SIZE, 36,
                        color=bg, radius=10, outline=True, outline_color=accent)
            title = "GHOST CAPTURED" if self.session.captured else "GHOST ESCAPED"
            gr, gc = self.session.ghost_position
            sub = f"{self.session.turns} turns  ·  {self.session.distance_traveled} moves  ·  Ghost R{gr + 1:02} C{gc + 1:02}"
            self._text(title, self.BOARD_PANEL_X + 14, 134, font="section", color=accent)
            self._text(sub, self.BOARD_PANEL_X + self.BOARD_PANEL_SIZE - 14, 137,
                       font="small", color=self.TEXT, align="right")
        else:
            self._text("LIVE GRID", 52, 136, font="section")
            self._text(f"{self.grid_size} x {self.grid_size}  /  SEED {self.seed}",
                       632, 141, font="small", color=self.MUTED, align="right")

    def _draw_board(self) -> None:
        self._panel(self.BOARD_PANEL_X, self.BOARD_PANEL_Y,
                    self.BOARD_PANEL_SIZE, self.BOARD_PANEL_SIZE,
                    color=self.SURFACE, outline=True)
        side = self.BOARD_SIZE / self.grid_size
        belief = self.session.visible_belief
        max_p = max(float(belief.max()), 1e-12)
        hr, hc = self.session.hunter_position
        compact_axes = self.grid_size > 12

        # Row and column coordinate labels aligned with grid cells.
        for idx in range(self.grid_size):
            col_cx = int(self.BOARD_X + (idx + 0.5) * side)
            row_cy = int(self.BOARD_Y + (idx + 0.5) * side)
            col_label = f"{idx + 1}" if compact_axes else f"C{idx + 1}"
            row_label = f"{idx + 1}" if compact_axes else f"R{idx + 1}"
            self._text(col_label, col_cx, self.BOARD_Y - 15, font="mini",
                       color=self.MINT if idx == hc else self.MUTED, align="center")
            self._text(row_label, self.BOARD_X - 18, row_cy, font="mini",
                       color=self.MINT if idx == hr else self.MUTED, align="center")

        ghost_pos = self.session.ghost_position if self.session.finished else None
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(round(self.BOARD_X + c * side),
                                   round(self.BOARD_Y + r * side),
                                   round((c + 1) * side) - round(c * side) - 1,
                                   round((r + 1) * side) - round(r * side) - 1)
                probability = float(belief[cell_index((r, c), self.grid_size)])
                color = pygame.Color(*self.BOARD_BG)
                if self.heatmap_enabled:
                    scale = (probability / max_p) ** 0.65
                    color = color.lerp(pygame.Color(*self.MINT), min(0.82, scale * 0.68))
                if ghost_pos == (r, c):
                    color = color.lerp(pygame.Color(*self.DANGER), 0.32)
                pygame.draw.rect(self.screen, color, rect, border_radius=2)
                border_col = self.DANGER if ghost_pos == (r, c) else (40, 66, 73)
                border_w = 2 if ghost_pos == (r, c) else 1
                pygame.draw.rect(self.screen, border_col, rect, width=border_w, border_radius=2)
                if (self.heatmap_enabled and side >= 40 and probability >= 0.012
                        and (r, c) != (hr, hc) and (r, c) != ghost_pos):
                    txt_color = (9, 26, 31) if probability / max_p > 0.45 else self.MUTED
                    label = f"{probability * 100:.0f}%"
                    self._text(label, rect.x + 5, rect.y + 4, font="mini", color=txt_color)

        center = (int(self.BOARD_X + (hc + 0.5) * side),
                  int(self.BOARD_Y + (hr + 0.5) * side))
        radius = max(8, int(side * 0.29))
        pygame.draw.circle(self.screen, (7, 24, 29), center, radius + 5)
        pygame.draw.circle(self.screen, self.LIGHT, center, radius, width=3)
        pygame.draw.circle(self.screen, self.LIGHT, center, 4)
        pygame.draw.line(self.screen, self.LIGHT,
                         (center[0] - radius - 6, center[1]),
                         (center[0] - radius + 1, center[1]), 2)
        pygame.draw.line(self.screen, self.LIGHT,
                         (center[0] + radius - 1, center[1]),
                         (center[0] + radius + 6, center[1]), 2)

        if ghost_pos is not None:
            gr, gc = ghost_pos
            ghost_center = (int(self.BOARD_X + (gc + 0.5) * side),
                            int(self.BOARD_Y + (gr + 0.5) * side))
            if ghost_pos == (hr, hc):
                # Captured on the hunter's cell: draw outer ring + corner badge so both stay readable.
                pygame.draw.circle(self.screen, self.DANGER, ghost_center, radius + 7, width=3)
                badge_x = round(self.BOARD_X + (gc + 1) * side) - 11
                badge_y = round(self.BOARD_Y + gr * side) + 10
                pygame.draw.circle(self.screen, self.DANGER, (badge_x, badge_y), 8)
                self._text("G", badge_x, badge_y, font="mini", color=self.BG, align="center")
            else:
                pygame.draw.circle(self.screen, (42, 18, 20), ghost_center, radius + 4)
                pygame.draw.circle(self.screen, self.DANGER, ghost_center, radius + 3, width=3)
                self._text("G", ghost_center[0], ghost_center[1], font="mono",
                           color=self.DANGER, align="center")

    def _draw_below_board(self) -> None:
        self._text("HUNTER", 54, 766, font="mini", color=self.MUTED)
        pygame.draw.circle(self.screen, self.LIGHT, (120, 774), 7, width=2)
        self._text("BELIEF", 160, 766, font="mini", color=self.MUTED)
        pygame.draw.rect(self.screen, self.MINT, (212, 768, 12, 12), border_radius=2)
        if self.session.finished:
            self._text("GHOST", 252, 766, font="mini", color=self.MUTED)
            pygame.draw.circle(self.screen, self.DANGER, (310, 774), 7, width=2)
            self._text("GHOST POSITION REVEALED", 632, 766, font="mini",
                       color=self.MINT if self.session.captured else self.DANGER,
                       align="right")
        else:
            self._text("GHOST HIDDEN UNTIL GAME ENDS", 632, 766, font="mini",
                       color=self.MUTED, align="right")

    def _draw_info(self) -> None:
        x, w = self.PANEL_X, self.PANEL_W

        # Panel 1: Sensor observation & live accuracy selector
        self._panel(x, 127, w, 148, outline=True)
        self._text("SENSOR  /  CURRENT OBSERVATION", x + 20, 141,
                   font="small", color=self.MUTED)
        self._text("NOISE: -2 / -1 / 0 / +1 / +2", x + w - 20, 143,
                   font="mini", color=self.MUTED, align="right")
        reading_rect = self._text(f"{self.session.reading:02d}", x + 20, 170,
                                  font="display", color=self.MINT)
        self._text("CELLS", reading_rect.right + 12, 188, font="small", color=self.MUTED)
        self._text("NOISY MANHATTAN DISTANCE", x + 20, 232,
                   font="mini", color=self.MUTED)
        self._text(f"EXACT ACCURACY ({self.sensor_accuracy * 100:.0f}%)", x + w - 20, 173,
                   font="mini", color=self.MUTED, align="right")
        for i, acc in enumerate(self.ACCURACY_PRESETS):
            bx = x + 239 + i * 66
            self._button(f"acc_{acc:.2f}", f"{acc * 100:.0f}%", (bx, 196, 60, 32),
                         active=abs(self.sensor_accuracy - acc) < 1e-6)
        self._text("Click to switch sensor noise & restart", x + w - 20, 238,
                   font="mini", color=self.MUTED, align="right")

        # Panel 2: Belief posterior + entropy
        self._panel(x, 285, w, 114, outline=True)
        name = "HMM POSTERIOR" if self.session.mode != "baseline" else "SENSOR-ONLY ESTIMATE"
        self._text(name, x + 20, 298, font="small", color=self.MUTED)
        entropy = self.session.history[-1].entropy if self.session.history else 0.0
        self._text(f"ENTROPY  {entropy:.2f} BITS", x + w - 20, 300,
                   font="mini", color=self.MUTED, align="right")
        belief = self.session.visible_belief
        top_idx = int(belief.argmax())
        likely = position_for(top_idx, self.grid_size)
        self._text(f"R{likely[0] + 1:02}  ·  C{likely[1] + 1:02}", x + 20, 326, font="title")
        self._text(f"{belief[top_idx] * 100:.1f}%", x + w - 20, 326,
                   font="title", color=self.MINT, align="right")
        track = pygame.Rect(x + 20, 376, w - 40, 6)
        pygame.draw.rect(self.screen, self.OUTLINE, track, border_radius=3)
        fill = track.copy()
        fill.width = max(2, int(track.width * float(belief[top_idx])))
        pygame.draw.rect(self.screen, self.MINT, fill, border_radius=3)

        # Panel 3: Episode status
        self._panel(x, 409, w, 98, outline=True)
        self._text("EPISODE STATUS", x + 20, 421, font="small", color=self.MUTED)
        stat_items = (("TURNS", f"{self.session.turns:02} / {self.max_turns}"),
                      ("MOVES", str(self.session.distance_traveled)),
                      ("ATTEMPTS", str(self.session.capture_attempts)))
        for i, (label, value) in enumerate(stat_items):
            ix = x + 22 + i * 165
            self._text(label, ix, 446, font="mini", color=self.MUTED)
            self._text(value, ix, 466, font="section")
        pygame.draw.line(self.screen, self.OUTLINE, (x + 178, 444), (x + 178, 494), 1)
        pygame.draw.line(self.screen, self.OUTLINE, (x + 344, 444), (x + 344, 494), 1)

        # Panel 4: Control center + speed bar / capture button
        self._panel(x, 517, w, 186, outline=True)
        self._text("CONTROL CENTER", x + 20, 529, font="small", color=self.MUTED)
        choices = (("human", "1   HUMAN"), ("ai", "2   HMM AI"),
                   ("baseline", "3   BASELINE"))
        for i, (key, label) in enumerate(choices):
            bx = x + 18 + i * 163
            self._button(key, label, (bx, 556, 155, 36), active=self.session.mode == key)
        if self.session.mode == "human":
            self._text("MOVE: WASD / ARROW KEYS", x + 20, 616, font="small")
            self._button("capture", "C   ATTEMPT CAPTURE", (x + 278, 605, 221, 38),
                         strong=not self.session.finished)
            assist_msg = (
                "Heatmap assistance ON  ·  press H to play unassisted"
                if self.heatmap_enabled
                else "Heatmap OFF  ·  unassisted human play"
            )
            self._text(assist_msg, x + 20, 662, font="mini", color=self.MUTED)
        else:
            self._button("step", "SPACE   STEP", (x + 18, 605, 234, 38))
            name = "PAUSE" if self.autoplay else "A   AUTOPLAY"
            self._button("autoplay", name, (x + 265, 605, 234, 38), strong=self.autoplay)
            self._text("SPEED  ([ / ])", x + 20, 663, font="mini", color=self.MUTED)
            self._button("speed_slower", "- SLOWER", (x + 220, 654, 88, 32))
            self._text(f"{self.auto_interval} ms", x + 358, 670,
                       font="small", color=self.TEXT, align="center")
            self._button("speed_faster", "+ FASTER", (x + 411, 654, 88, 32))

        # Panel 5: Utility bar
        self._panel(x, 713, w, 71, outline=True)
        self._button("heatmap", "H   HEATMAP  " + ("ON" if self.heatmap_enabled else "OFF"),
                     (x + 16, 727, 186, 43), active=self.heatmap_enabled)
        self._button("restart", "R   RESTART", (x + 212, 727, 138, 43))
        self._button("new_seed", "N   NEW SEED", (x + 360, 727, 141, 43))

    def _draw_footer(self) -> None:
        pygame.draw.line(self.screen, self.OUTLINE, (51, 802), (1185, 802), 1)
        message = self.status
        if len(message) > 95:
            message = message[:92] + "..."
        self._text(message, 52, 811, font="small", color=self.MUTED)
        self._text("P  SAVE SCREENSHOT    ·    ESC  QUIT", 1185, 812,
                   font="mini", color=self.MUTED, align="right")

    def _human_row(self, label: str, items: list[dict]) -> tuple[str, str, str, str, str]:
        completed = [item for item in items if item.get("captured")]
        count = len(items)
        return (
            label,
            str(count),
            f"{100 * len(completed) / count:.1f}%" if count else "—",
            f"{sum(r['turns'] for r in completed) / len(completed):.1f}" if completed else "—",
            f"{sum(r['distance'] for r in completed) / len(completed):.1f}" if completed else "—",
        )

    def _draw_scoreboard(self) -> None:
        backdrop = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        backdrop.fill((4, 15, 21, 224))
        self.screen.blit(backdrop, (0, 0))
        x, y, w, h = 150, 118, 950, 594
        self._modal_rect = pygame.Rect(x, y, w, h)
        self._panel(x, y, w, h, color=self.SURFACE, outline=True)
        self._text("HUMAN VS AI  /  PERFORMANCE", x + 30, y + 24, font="title")
        self._button("close_scores", "TAB   CLOSE", (x + w - 160, y + 22, 130, 34), modal=True)
        self._text("Capture rate, turns, and movement — means are computed over successful captures.",
                   x + 30, y + 72, font="small", color=self.MUTED)
        cols = [x + 32, x + 318, x + 452, x + 622, x + 784]
        header = ["STRATEGY", "EPISODES", "CAPTURE %", "AVG TURNS", "AVG MOVES"]
        for col, label in zip(cols, header):
            self._text(label, col, y + 114, font="mini", color=self.MUTED)
        pygame.draw.line(self.screen, self.OUTLINE, (x + 30, y + 142),
                         (x + w - 30, y + 142), 1)
        rows = []
        try:
            p = self._human_score_path()
            human = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
            unassisted = [item for item in human if item.get("heatmap_used") is False]
            assisted = [item for item in human if item.get("heatmap_used") is not False]
            rows.append(self._human_row("Human (no heatmap)", unassisted))
            rows.append(self._human_row("Human (+ heatmap)", assisted))
        except (OSError, ValueError, KeyError):
            rows.extend((("Human (no heatmap)", "—", "—", "—", "—"),
                         ("Human (+ heatmap)", "—", "—", "—", "—")))
        try:
            p = PROJECT_ROOT / "results" / "summary.json"
            summary = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            for mode, label in (("ai", "HMM-guided AI"), ("baseline", "Sensor-only")):
                info = summary.get(mode, {})
                rows.append((label, str(info.get("episodes", "—")),
                             f"{info['capture_rate_pct']:.1f}%" if info.get("capture_rate_pct") is not None else "—",
                             f"{info['mean_turns_to_capture']:.1f}" if info.get("mean_turns_to_capture") is not None else "—",
                             f"{info['mean_movement_to_capture']:.1f}" if info.get("mean_movement_to_capture") is not None else "—"))
        except (OSError, ValueError, TypeError):
            rows.extend((("HMM-guided AI", "—", "—", "—", "—"),
                         ("Sensor-only", "—", "—", "—", "—")))
        for row_idx, values in enumerate(rows):
            ry = y + 162 + row_idx * 68
            if row_idx % 2 == 0:
                self._panel(x + 23, ry - 12, w - 46, 52, color=self.SURFACE2, radius=8)
            for ci, (col, value) in enumerate(zip(cols, values)):
                self._text(value, col, ry, font="body",
                           color=self.MINT if ci == 2 else self.TEXT)
        self._text("Human scores track whether the HMM heatmap was used (legacy records default to + heatmap).",
                   x + 30, y + 458, font="small", color=self.MUTED)
        self._text("AI results come from 100 paired seeded episodes; human play is not a controlled pair.",
                   x + 30, y + 482, font="small", color=self.MUTED)
        self._text("Run python run_evaluation.py to refresh the AI comparison. Press TAB or ESC to close.",
                   x + 30, y + 522, font="small", color=self.LIGHT)

    def draw(self) -> None:
        self.hitboxes.clear()
        self._modal_rect = None
        self.screen.fill(self.BG)
        self._draw_header()
        self._draw_board()
        self._draw_below_board()
        self._draw_info()
        self._draw_footer()
        if self.show_scores:
            self._draw_scoreboard()
        viewport = self._viewport()
        self.window.fill((7, 15, 20))
        if (viewport.width, viewport.height) == (self.WIDTH, self.HEIGHT):
            self.window.blit(self.screen, viewport.topleft)
        else:
            scaled = pygame.transform.smoothscale(self.screen, viewport.size)
            self.window.blit(scaled, viewport.topleft)
        pygame.display.flip()

    def run(self) -> None:
        try:
            while self.running:
                self.handle_events()
                now = pygame.time.get_ticks()
                if (self.autoplay and not self.show_scores
                        and self.session.mode != "human"
                        and not self.session.finished
                        and now - self.last_ai_tick >= self.auto_interval):
                    self._ai_turn()
                    self.last_ai_tick = now
                self.draw()
                self.clock.tick(60)
        finally:
            pygame.quit()
