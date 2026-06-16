"""Persistent fog-of-war map of everything the agent has ever seen.

The model has no spatial memory: each turn it only sees the current screen,
so left alone it re-explores rooms and circles mazes for hours. The WorldMap
accumulates the walkability of every tile that has appeared on screen, per
game map, in the game's own coordinate system, and turns that into the three
things the agent needs:

- a text rendering of the explored map with coordinate rulers
- the "frontier": reachable tiles bordering unexplored space (the single
  most effective anti-stuck signal reported by Gemini Plays Pokemon)
- A* paths between any two known coordinates, so navigate_to can reach
  places far beyond the visible screen

Coordinates everywhere are the game's (x, y) walking-tile coordinates as
read from RAM: x grows rightward, y grows downward. Tiles seen past a map's
edge (outdoor connection strips) are kept in the current map's frame; paths
to them simply walk off the edge, which is how map transitions happen.
"""

import heapq
import json
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

WALKABLE = "."
WALL = "#"
UNKNOWN = "?"

# The visible screen is a 9x10 grid of walking tiles with the player fixed
# at its center cell.
SCREEN_ROWS = 9
SCREEN_COLS = 10
PLAYER_ROW = 4
PLAYER_COL = 4

DIRECTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

FACING_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}

# Cap the rendered window so huge maps can't blow up the prompt.
RENDER_RADIUS = 30
FRONTIER_LIMIT = 10


class WorldMap:
    """Per-map dictionary of every tile ever seen, with text/path queries."""

    def __init__(self):
        self._lock = threading.Lock()
        self._maps: dict[int, dict[tuple[int, int], str]] = {}

    # ------------------------------------------------------------------
    # Building the map
    # ------------------------------------------------------------------

    def update_from_screen(self, map_id, player_pos, terrain, sprites) -> None:
        """Record the currently visible screen into the map.

        Args:
            map_id: Current map number (RAM 0xD35E).
            player_pos: The player's global (x, y) from RAM.
            terrain: 9x10 downsampled collision grid (0 = wall).
            sprites: Set of (col, row) screen cells occupied by sprites —
                skipped, because the tile underneath an NPC is unknowable
                and NPCs move.
        """
        px, py = player_pos
        with self._lock:
            tiles = self._maps.setdefault(map_id, {})
            for row in range(SCREEN_ROWS):
                for col in range(SCREEN_COLS):
                    if (col, row) in sprites:
                        continue
                    pos = (px + col - PLAYER_COL, py + row - PLAYER_ROW)
                    tiles[pos] = WALKABLE if terrain[row][col] else WALL
            # The player's own cell reads as a sprite; it is proven walkable
            tiles[(px, py)] = WALKABLE

    def _tiles(self, map_id) -> dict:
        with self._lock:
            return dict(self._maps.get(map_id, {}))

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def find_path(
        self, map_id, start, goal, blocked=frozenset()
    ) -> tuple[list[str], bool]:
        """A* over known-walkable tiles from start to goal.

        The goal itself may be a wall, an NPC, or unexplored: the path then
        ends with one step "into" it (a bump — how you face/talk to things).

        Args:
            blocked: Set of (position, direction) edges to avoid — filled by
                the navigator when a step that should work doesn't (moving
                NPCs, ledges, tile quirks the collision data doesn't show).

        Returns:
            (directions, bump): the button sequence, and whether its last
            step targets a non-walkable tile. ([], False) when unreachable.
        """
        tiles = self._tiles(map_id)
        tiles[start] = WALKABLE
        bump = tiles.get(goal) != WALKABLE

        def neighbors(pos):
            for direction, (dx, dy) in DIRECTION_DELTAS.items():
                if (pos, direction) in blocked:
                    continue
                yield (pos[0] + dx, pos[1] + dy), direction

        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        open_set = [(heuristic(start), 0, start)]
        came_from: dict = {}
        g_score = {start: 0}
        counter = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)
            arrived = current == goal or (bump and heuristic(current) == 1)
            if arrived:
                path = []
                pos = current
                while pos in came_from:
                    pos, direction = came_from[pos]
                    path.append(direction)
                path.reverse()
                if bump and current != goal:
                    dx, dy = goal[0] - current[0], goal[1] - current[1]
                    for direction, delta in DIRECTION_DELTAS.items():
                        if delta == (dx, dy):
                            path.append(direction)
                return path, bump
            for neighbor, direction in neighbors(current):
                if tiles.get(neighbor) != WALKABLE and neighbor != goal:
                    continue
                if neighbor == goal and bump:
                    continue  # bump goals are entered via the arrival check
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = (current, direction)
                    g_score[neighbor] = tentative
                    counter += 1
                    heapq.heappush(
                        open_set, (tentative + heuristic(neighbor), counter, neighbor)
                    )
        return [], False

    def frontier(
        self, map_id, start, limit=FRONTIER_LIMIT, ignore=frozenset()
    ) -> list[tuple[int, tuple]]:
        """Reachable explored tiles that border unexplored space, nearest first.

        Args:
            ignore: Positions whose unknown-ness doesn't count — the tiles
                currently hidden under on-screen NPCs, so "go stand next to
                that person" isn't sold as exploration.

        Returns a list of (steps_away, (x, y)).
        """
        tiles = self._tiles(map_id)
        tiles[start] = WALKABLE
        found = []
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            edges_unknown = False
            for dx, dy in DIRECTION_DELTAS.values():
                neighbor = (pos[0] + dx, pos[1] + dy)
                if neighbor not in tiles:
                    if neighbor not in ignore:
                        edges_unknown = True
                elif tiles[neighbor] == WALKABLE and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, dist + 1))
            if edges_unknown and dist > 0:
                found.append((dist, pos))
        found.sort()
        return found[:limit]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self, map_id, location, player_pos, facing, warps=(), sprites=()
    ) -> str:
        """Text view of the explored map, plus door/frontier coordinate lists."""
        tiles = self._tiles(map_id)
        px, py = player_pos
        tiles.setdefault((px, py), WALKABLE)

        xs = [x for x, _ in tiles]
        ys = [y for _, y in tiles]
        x0 = max(min(xs), px - RENDER_RADIUS)
        x1 = min(max(xs), px + RENDER_RADIUS)
        y0 = max(min(ys), py - RENDER_RADIUS)
        y1 = min(max(ys), py + RENDER_RADIUS)

        sprites = set(sprites)
        warp_set = set(warps)

        label_width = max(len(f"y={y}") for y in (y0, y1))
        ruler = [" "] * (x1 - x0 + 1)
        for x in range(x0, x1 + 1):
            # Only place labels that fit entirely (a clipped "10" reads as "1")
            if x % 5 == 0 and x - x0 + len(str(x)) <= len(ruler):
                for i, ch in enumerate(str(x)):
                    ruler[x - x0 + i] = ch
        lines = [
            f"{location} — explored map. You are at (x={px}, y={py}), facing {facing}.",
            " " * (label_width + 1) + "".join(ruler),
        ]
        for y in range(y0, y1 + 1):
            row = []
            for x in range(x0, x1 + 1):
                if (x, y) == (px, py):
                    row.append(FACING_ARROWS.get(facing, "P"))
                elif (x, y) in sprites:
                    row.append("S")
                elif (x, y) in warp_set:
                    row.append("D")
                else:
                    row.append(tiles.get((x, y), UNKNOWN))
            lines.append(f"{f'y={y}':>{label_width}} " + "".join(row))

        lines.append(
            "Legend: . explored ground, # wall/obstacle, ? unexplored, "
            "D door/warp (walk onto it to go through), S person/object "
            "standing there right now, arrow = you."
        )
        if warps:
            coords = ", ".join(f"({x}, {y})" for x, y in warps)
            lines.append(f"Doors/warps on this map: {coords}")
        if sprites:
            coords = ", ".join(f"({x}, {y})" for x, y in sorted(sprites))
            lines.append(f"People/objects on screen: {coords}")

        frontier = self.frontier(map_id, player_pos, ignore=sprites)
        if frontier:
            spots = ", ".join(f"({x}, {y}) {d} steps" for d, (x, y) in frontier)
            lines.append(f"Reachable UNEXPLORED spots: {spots}")
        else:
            lines.append(
                "No unexplored spots left here — progress is through a "
                "door/warp or off the map edge."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence (saved alongside the numbered game states)
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        with self._lock:
            data = {
                str(map_id): {f"{x},{y}": ch for (x, y), ch in tiles.items()}
                for map_id, tiles in self._maps.items()
            }
        with open(path, "w") as f:
            json.dump({"maps": data}, f)

    def load(self, path) -> None:
        with open(path) as f:
            data = json.load(f)
        maps = {}
        for map_id, tiles in data.get("maps", {}).items():
            maps[int(map_id)] = {
                tuple(int(n) for n in key.split(",")): ch
                for key, ch in tiles.items()
            }
        with self._lock:
            self._maps = maps
        logger.info(f"[WorldMap] Loaded explored map from {path}")
