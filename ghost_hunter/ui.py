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
from .scores import InvalidScoreHistory, append_score, read_scores


class GameUI:
    """Compact, keyboard-friendly desktop UI with no external image assets."""

    WIDTH, HEIGHT = 1250, 840
    BOARD_X, BOARD_Y, BOARD_SIZE = 64, 182, 555
    PANEL_X, PANEL_W = 672, 513
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
        self.show_scores = False
        self.autoplay = False
        self.auto_interval = 300
        self.last_ai_tick = 0
        self.running = True
        self.status = "The ghost is hidden. Use the noisy reading to track it."
        self.hitboxes: dict[str, pygame.Rect] = {}
        self._result_saved = False
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
               outline=False) -> None:
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, color or self.SURFACE, rect, border_radius=radius)
        if outline:
            pygame.draw.rect(self.screen, self.OUTLINE, rect, width=1, border_radius=radius)

    def _button(self, key: str, label: str, rect: tuple[int, int, int, int], *,
                active: bool = False, strong: bool = False) -> None:
        area = pygame.Rect(*rect)
        self.hitboxes[key] = area
        hovered = area.collidepoint(self._to_logical(pygame.mouse.get_pos()))
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
        self.autoplay = False
        self._result_saved = False
        self.status = f"New {self._mode_label()} episode. Ghost position is hidden."

    def _mode_label(self) -> str:
        return {"human": "Human", "ai": "HMM AI", "baseline": "Sensor-only"}[self.session.mode]

    def _human_score_path(self) -> Path:
        return PROJECT_ROOT / "results" / "human_scores.json"

    def _save_human_score(self) -> None:
        if self._result_saved or not self.session.finished or self.session.mode != "human":
            return
        record = {"date": datetime.now().isoformat(timespec="seconds"),
                  "seed": self.seed, **self.session.result()}
        try:
            backup = append_score(self._human_score_path(), record)
        except OSError:
            self.status = "Could not save this score. Check the results folder permissions."
            return
        self._result_saved = True
        if backup is not None:
            self.status = f"Score saved. Previous score file backed up as {backup.name}."

    def take_turn(self, action: str) -> None:
        if self.session.finished:
            return
        if not self.session.take_action(action):
            self.status = "Invalid move: board edge reached."
            return
        if self.session.captured:
            self.status = f"Captured in {self.session.turns} turns!"
        elif self.session.finished:
            self.status = "Turn limit reached. The ghost escaped."
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
        action = HunterAgent(self.session.mode).choose_action(self.session)
        self.take_turn(action)

    def _screenshot(self) -> None:
        folder = PROJECT_ROOT / "results" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"ghost_hunter_{self.session.mode}_seed{self.seed}_turn{self.session.turns}.png"
        pygame.image.save(self.screen, str(path))
        self.status = f"Screenshot saved: {path.name}"

    def _click(self, key: str) -> None:
        if key in ("human", "ai", "baseline"):
            self.new_game(key)
        elif key == "heatmap":
            self.heatmap_enabled = not self.heatmap_enabled
        elif key == "restart":
            self.new_game()
        elif key == "new_seed":
            self.new_game(next_seed=True)
        elif key == "scores":
            self.show_scores = not self.show_scores
        elif key == "step":
            self._ai_turn()
        elif key == "autoplay":
            if self.session.mode != "human" and not self.session.finished:
                self.autoplay = not self.autoplay
                self.last_ai_tick = pygame.time.get_ticks()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for key, rect in tuple(self.hitboxes.items()):
                    if rect.collidepoint(self._to_logical(event.pos)):
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
                    self.heatmap_enabled = not self.heatmap_enabled
                elif event.key == pygame.K_p:
                    self._screenshot()
                elif event.key == pygame.K_TAB:
                    self.show_scores = not self.show_scores
                elif event.key == pygame.K_SPACE:
                    self._ai_turn()
                elif event.key == pygame.K_a and self.session.mode != "human":
                    if not self.session.finished:
                        self.autoplay = not self.autoplay
                        self.last_ai_tick = pygame.time.get_ticks()
                elif event.key == pygame.K_LEFTBRACKET:
                    self.auto_interval = min(1000, self.auto_interval + 50)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.auto_interval = max(50, self.auto_interval - 50)
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
        title = self._text("GHOST", 64, 34, font="display", color=self.TEXT)
        self._text("/ HUNTER", title.right + 6, 34, font="display", color=self.MINT)
        self._text("PROBABILISTIC TRACKING LAB", 66, 89, font="small", color=self.MUTED)
        self._text("HIDDEN MARKOV MODEL  /  FORWARD INFERENCE", 1185, 47,
                   font="small", color=self.MUTED, align="right")
        self._button("scores", "TAB   SCOREBOARD", (929, 81, 256, 30))
        pygame.draw.line(self.screen, self.OUTLINE, (64, 116), (1185, 116), 1)
        self._text("LIVE GRID", 64, 143, font="section")
        self._text(f"{self.grid_size} x {self.grid_size}  /  SEED {self.seed}",
                   618, 148, font="small", color=self.MUTED, align="right")

    def _draw_board(self) -> None:
        self._panel(self.BOARD_X - 13, self.BOARD_Y - 13,
                    self.BOARD_SIZE + 26, self.BOARD_SIZE + 26,
                    color=self.SURFACE, outline=True)
        side = self.BOARD_SIZE / self.grid_size
        belief = self.session.visible_belief
        max_p = max(float(belief.max()), 1e-12)
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
                pygame.draw.rect(self.screen, color, rect, border_radius=2)
                # Keep the grid readable even without heatmap.
                pygame.draw.rect(self.screen, (40, 66, 73), rect, width=1, border_radius=2)
                if self.heatmap_enabled and side >= 40 and probability >= 0.012:
                    txt_color = (9, 26, 31) if probability / max_p > 0.45 else self.MUTED
                    label = f"{probability * 100:.0f}%"
                    self._text(label, rect.x + 6, rect.y + 5, font="mini", color=txt_color)

        hr, hc = self.session.hunter_position
        center = (int(self.BOARD_X + (hc + .5) * side),
                  int(self.BOARD_Y + (hr + .5) * side))
        radius = max(8, int(side * .29))
        pygame.draw.circle(self.screen, (7, 24, 29), center, radius + 6)
        pygame.draw.circle(self.screen, self.LIGHT, center, radius, width=3)
        pygame.draw.circle(self.screen, self.LIGHT, center, 4)
        pygame.draw.line(self.screen, self.LIGHT,
                         (center[0] - radius - 6, center[1]),
                         (center[0] - radius + 1, center[1]), 2)
        pygame.draw.line(self.screen, self.LIGHT,
                         (center[0] + radius - 1, center[1]),
                         (center[0] + radius + 6, center[1]), 2)
        if self.session.finished:
            gr, gc = self.session.ghost_position
            ghost_center = (int(self.BOARD_X + (gc + .5) * side),
                            int(self.BOARD_Y + (gr + .5) * side))
            pygame.draw.circle(self.screen, self.DANGER, ghost_center, radius + 3, width=4)
            self._text("G", ghost_center[0], ghost_center[1], font="mono",
                       color=self.DANGER, align="center")
        if self.session.finished:
            title = "GHOST CAPTURED" if self.session.captured else "GHOST ESCAPED"
            sub = f"{self.session.turns} turns  /  {self.session.distance_traveled} moves"
            box = pygame.Surface((440, 92), pygame.SRCALPHA)
            pygame.draw.rect(box, (12, 26, 33, 235), box.get_rect(), border_radius=12)
            self.screen.blit(box, (self.BOARD_X + (self.BOARD_SIZE - 440) // 2,
                                   self.BOARD_Y + self.BOARD_SIZE // 2 - 46))
            self._text(title, self.BOARD_X + self.BOARD_SIZE // 2,
                       self.BOARD_Y + self.BOARD_SIZE // 2 - 15,
                       font="title", color=self.MINT if self.session.captured else self.DANGER,
                       align="center")
            self._text(sub, self.BOARD_X + self.BOARD_SIZE // 2,
                       self.BOARD_Y + self.BOARD_SIZE // 2 + 19,
                       font="small", color=self.TEXT, align="center")

    def _draw_below_board(self) -> None:
        self._text("HUNTER", 65, 766, font="mini", color=self.MUTED)
        pygame.draw.circle(self.screen, self.LIGHT, (132, 775), 7, width=2)
        self._text("BELIEF", 184, 766, font="mini", color=self.MUTED)
        pygame.draw.rect(self.screen, self.MINT, (236, 768, 12, 12), border_radius=2)
        self._text("GHOST HIDDEN UNTIL GAME ENDS", 618, 767, font="mini",
                   color=self.MUTED, align="right")

    def _draw_info(self) -> None:
        x, w = self.PANEL_X, self.PANEL_W
        self._panel(x, 132, w, 140, outline=True)
        self._text("SENSOR  /  CURRENT OBSERVATION", x + 22, 149,
                   font="small", color=self.MUTED)
        self._text(f"{self.session.reading:02d}", x + 20, 178, font="display", color=self.MINT)
        self._text("CELLS", x + 108, 191, font="small", color=self.MUTED)
        self._text("NOISY MANHATTAN DISTANCE", x + 22, 239,
                   font="mini", color=self.MUTED)
        self._text(f"Zero noise: {self.sensor_accuracy * 100:.0f}%", x + w - 20, 192,
                   font="small", color=self.TEXT, align="right")
        self._text("Noise: -2 / -1 / 0 / +1 / +2", x + w - 20, 233,
                   font="mini", color=self.MUTED, align="right")

        self._panel(x, 284, w, 118, outline=True)
        name = "HMM POSTERIOR" if self.session.mode != "baseline" else "SENSOR-ONLY ESTIMATE"
        self._text(name, x + 22, 298, font="small", color=self.MUTED)
        belief = self.session.visible_belief
        top_idx = int(belief.argmax())
        likely = position_for(top_idx, self.grid_size)
        self._text(f"R{likely[0] + 1:02}  ·  C{likely[1] + 1:02}", x + 22, 329, font="title")
        self._text(f"{belief[top_idx] * 100:.1f}%", x + w - 22, 329,
                   font="title", color=self.MINT, align="right")
        track = pygame.Rect(x + 22, 380, w - 44, 5)
        pygame.draw.rect(self.screen, self.OUTLINE, track, border_radius=3)
        fill = track.copy()
        fill.width = max(2, int(track.width * float(belief[top_idx])))
        pygame.draw.rect(self.screen, self.MINT, fill, border_radius=3)

        self._panel(x, 414, w, 112, outline=True)
        self._text("EPISODE STATUS", x + 22, 429, font="small", color=self.MUTED)
        stat_items = (("TURNS", f"{self.session.turns:02} / {self.max_turns}"),
                      ("MOVES", str(self.session.distance_traveled)),
                      ("ATTEMPTS", str(self.session.capture_attempts)))
        for i, (label, value) in enumerate(stat_items):
            ix = x + 24 + i * 162
            self._text(label, ix, 457, font="mini", color=self.MUTED)
            self._text(value, ix, 479, font="section")
        pygame.draw.line(self.screen, self.OUTLINE, (x + 184, 454), (x + 184, 512), 1)
        pygame.draw.line(self.screen, self.OUTLINE, (x + 346, 454), (x + 346, 512), 1)

        self._panel(x, 538, w, 158, outline=True)
        self._text("CONTROL CENTER", x + 22, 551, font="small", color=self.MUTED)
        choices = (("human", "1   HUMAN"), ("ai", "2   HMM AI"),
                   ("baseline", "3   BASELINE"))
        for i, (key, label) in enumerate(choices):
            bx = x + 18 + i * 161
            self._button(key, label, (bx, 585, 154, 37), active=self.session.mode == key)
        if self.session.mode == "human":
            self._text("MOVE: WASD / ARROWS", x + 22, 642, font="small")
            self._text("C  ATTEMPT CAPTURE", x + 261, 642, font="small")
        else:
            self._button("step", "SPACE  STEP", (x + 18, 635, 230, 41))
            name = "PAUSE" if self.autoplay else "A   AUTOPLAY"
            self._button("autoplay", name, (x + 263, 635, 230, 41), strong=self.autoplay)

        self._panel(x, 708, w, 76, outline=True)
        self._button("heatmap", "H   HEATMAP  " + ("ON" if self.heatmap_enabled else "OFF"),
                     (x + 14, 723, 180, 45), active=self.heatmap_enabled)
        self._button("restart", "R   RESTART", (x + 206, 723, 137, 45))
        self._button("new_seed", "N   NEW SEED", (x + 354, 723, 145, 45))

    def _draw_footer(self) -> None:
        pygame.draw.line(self.screen, self.OUTLINE, (64, 802), (1185, 802), 1)
        message = self.status
        if len(message) > 97:
            message = message[:94] + "..."
        self._text(message, 65, 811, font="small", color=self.MUTED)
        self._text("P  SAVE SCREENSHOT    ·    ESC  QUIT", 1184, 812,
                   font="mini", color=self.MUTED, align="right")

    def _draw_scoreboard(self) -> None:
        backdrop = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        backdrop.fill((4, 15, 21, 224))
        self.screen.blit(backdrop, (0, 0))
        x, y, w, h = 168, 156, 916, 515
        self._panel(x, y, w, h, color=self.SURFACE, outline=True)
        self._text("HUMAN VS AI  /  PERFORMANCE", x + 30, y + 25, font="title")
        self._text("Press TAB to return", x + w - 28, y + 33,
                   font="small", color=self.MUTED, align="right")
        self._text("Capture rate, turns, and movement — successful captures only for means.",
                   x + 30, y + 74, font="small", color=self.MUTED)
        cols = [x + 32, x + 277, x + 411, x + 590, x + 757]
        header = ["STRATEGY", "EPISODES", "CAPTURE %", "AVG TURNS", "AVG MOVES"]
        for col, label in zip(cols, header):
            self._text(label, col, y + 117, font="mini", color=self.MUTED)
        pygame.draw.line(self.screen, self.OUTLINE, (x + 30, y + 146),
                         (x + w - 30, y + 146), 1)
        rows = []
        score_error = ""
        try:
            human = read_scores(self._human_score_path())
            completed = [item for item in human if item.get("captured")]
            count = len(human)
            rows.append(("Human play", str(count),
                         f"{100 * len(completed) / count:.1f}%" if count else "—",
                         f"{sum(r['turns'] for r in completed) / len(completed):.1f}" if completed else "—",
                         f"{sum(r['distance'] for r in completed) / len(completed):.1f}" if completed else "—"))
        except (OSError, InvalidScoreHistory) as error:
            score_error = str(error)
            rows.append(("Human play", "—", "—", "—", "—"))
        try:
            p = PROJECT_ROOT / "results" / "summary.json"
            summary = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, ValueError):
            summary = {}
        for mode, label in (("ai", "HMM-guided AI"), ("baseline", "Sensor-only")):
            try:
                info = summary[mode]
                row = (label, str(info["episodes"]),
                       f"{info['capture_rate_pct']:.1f}%",
                       f"{info['mean_turns_to_capture']:.1f}" if info.get("mean_turns_to_capture") is not None else "—",
                       f"{info['mean_movement_to_capture']:.1f}" if info.get("mean_movement_to_capture") is not None else "—")
            except (KeyError, TypeError, ValueError):
                row = (label, "—", "—", "—", "—")
            rows.append(row)
        for row_idx, values in enumerate(rows):
            ry = y + 166 + row_idx * 72
            if row_idx % 2 == 0:
                self._panel(x + 23, ry - 13, w - 46, 53, color=self.SURFACE2, radius=8)
            for ci, (col, value) in enumerate(zip(cols, values)):
                self._text(value, col, ry, font="body",
                           color=self.MINT if ci == 2 else self.TEXT)
        if score_error:
            self._text(score_error[:84], x + 30, y + 416, font="small", color=self.DANGER)
        else:
            self._text("Human scores come from local play. The two AI strategies use paired",
                       x + 30, y + 416, font="small", color=self.MUTED)
            self._text("simulations. Human and AI games can use different settings.",
                       x + 30, y + 439, font="small", color=self.MUTED)
        self._text("Run python run_evaluation.py to refresh the 100-episode AI comparison.",
                   x + 30, y + 471, font="small", color=self.LIGHT)

    def draw(self) -> None:
        self.hitboxes.clear()
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
                if (self.autoplay and self.session.mode != "human"
                        and not self.session.finished
                        and now - self.last_ai_tick >= self.auto_interval):
                    self._ai_turn()
                    self.last_ai_tick = now
                self.draw()
                self.clock.tick(60)
        finally:
            pygame.quit()
