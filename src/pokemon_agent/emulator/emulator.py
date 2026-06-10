import concurrent.futures
import logging
import os
import threading
import heapq

from pokemon_agent.emulator.memory_reader import PokemonRedReader, StatusCondition
from PIL import Image
from pyboy import PyBoy

logger = logging.getLogger(__name__)

# Settle detection: after a button press, instead of waiting a fixed time, poll
# the game's own state until nothing is happening anymore (validated empirically
# against the old fixed wait: 3x faster on movement, and it stops capturing
# half-printed dialog text, which the fixed wait did ~16% of the time).
SETTLE_POLL_FRAMES = 8        # sample cadence
SETTLE_STABLE_SAMPLES = 4     # consecutive stable samples to settle (32 quiet frames)
SETTLE_STABLE_BLANK = 10      # required when the screen bottom is blank (scene
                              # transition gaps look idle before the next box opens)
SETTLE_TIMEOUT_FRAMES = 400   # ceiling: must exceed the longest legitimate wait
                              # (slow full-text dialog ~290 frames); only genuinely
                              # busy states (cutscene walks, battle intros) hit it
CONTINUE_ARROW_TILE = 0xEE    # the blinking ▼ — masked so its blink isn't "change"


class Emulator:
    """PyBoy wrapper where a single dedicated thread owns the emulator.

    SDL is not thread-safe: when ticks (which render the window) come from
    changing threads — LangGraph runs tools in worker threads — the window
    silently stops updating. So PyBoy is constructed and driven exclusively on
    one thread; every other thread submits work to it via _run_on_pyboy.
    """

    def __init__(self, rom_path, headless=True, sound=False):
        self._headless = headless
        self._stop_keepalive = threading.Event()
        self._keepalive_thread = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pyboy"
        )
        self._pyboy_thread = self._executor.submit(threading.current_thread).result()

        def construct():
            if headless:
                return PyBoy(rom_path, window="null", cgb=True)
            # XWayland tolerates threaded SDL rendering better than Wayland
            os.environ.setdefault("SDL_VIDEODRIVER", "x11")
            return PyBoy(rom_path, cgb=True, sound=sound)

        self.pyboy = self._executor.submit(construct).result()

    def _run_on_pyboy(self, fn, *args):
        """Run fn on the dedicated PyBoy thread (directly if already on it)."""
        if threading.current_thread() is self._pyboy_thread:
            return fn(*args)
        return self._executor.submit(fn, *args).result()

    def tick(self, frames):
        """Advance the emulator by the specified number of frames."""
        self._run_on_pyboy(self._tick_impl, frames)

    def _tick_impl(self, frames):
        for _ in range(frames):
            self.pyboy.tick()

    def initialize(self):
        """Initialize the emulator."""

        def boot():
            # Run the emulator for a short time to make sure it's ready
            self.pyboy.set_emulation_speed(0)
            self._tick_impl(3600)
            self.pyboy.set_emulation_speed(1)

        self._run_on_pyboy(boot)
        if not self._headless:
            # With a display, keep the game running while the agent thinks —
            # the window only renders and pumps events on tick().
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True
            )
            self._keepalive_thread.start()

    def _keepalive_loop(self):
        # Submits one frame at a time so agent work (button presses, reads)
        # interleaves fairly on the PyBoy thread. At emulation speed 1 each
        # tick self-paces to ~60fps.
        while not self._stop_keepalive.is_set():
            try:
                self._run_on_pyboy(self._tick_impl, 1)
            except RuntimeError:
                return  # executor shut down

    def get_screenshot(self):
        """Get the current screenshot."""

        def impl():
            # Copy: the screen buffer keeps updating as the game runs
            return Image.fromarray(self.pyboy.screen.ndarray.copy())

        return self._run_on_pyboy(impl)

    def load_state(self, state_filename):
        """
        Load a state from a pickled file into the emulator.
        The pickled file should contain a dictionary with a 'pyboy_state' key.
        
        Args:
            state_filename: Path to the state file
        """
        self._run_on_pyboy(lambda: self.pyboy.load_state(open(state_filename, "rb")))

    def save_state(self, state_filename):
        """Save the current emulator state to a file."""

        def impl():
            with open(state_filename, "wb") as f:
                self.pyboy.save_state(f)

        self._run_on_pyboy(impl)

    def press_buttons(self, buttons):
        """Press a sequence of buttons on the Game Boy, settling after each.

        Args:
            buttons (list[str]): List of buttons to press in sequence

        Returns:
            tuple[str, list[Image.Image]]: Result text and one keyframe per
            press, captured after that press settled (the last one is the
            current state). Lets the model see what happened inside a batch
            instead of only the end state.
        """
        def impl():
            results = []
            keyframes = []
            for button in buttons:
                if button not in ["a", "b", "start", "select", "up", "down", "left", "right"]:
                    results.append(f"Invalid button: {button}")
                    continue

                self.pyboy.button_press(button)
                self._tick_impl(10)   # Press briefly
                self.pyboy.button_release(button)
                self._settle_impl()   # Run until the game stops reacting

                keyframes.append(Image.fromarray(self.pyboy.screen.ndarray.copy()))
                results.append(f"Pressed {button}")
            return results, keyframes

        results, keyframes = self._run_on_pyboy(impl)
        return "\n".join(results), keyframes

    def _screen_mirror(self) -> bytes:
        """wTileMap with the blinking continue-arrow masked out."""
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_tilemap_buffer().replace(bytes([CONTINUE_ARROW_TILE]), b"\x00")

    @staticmethod
    def _bottom_blank(mirror: bytes) -> bool:
        """Whether the textbox area (bottom 6 tile rows) is blank."""
        bottom = mirror[12 * 20 :]
        blank = sum(1 for b in bottom if b in (0x7F, 0x00))
        return blank / len(bottom) > 0.9

    def _settle_impl(self) -> None:
        """Run the game until it settles after an input (on the PyBoy thread).

        Settled = the overworld engine reports idle AND the game's screen
        mirror stayed byte-identical across consecutive samples — meaning all
        text finished printing and all animations concluded. The timeout covers
        states that never go idle (cutscene walks, battle intros), where more
        waiting buys nothing: the press was a no-op and the next observation
        shows the still-busy screen.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        frames = 0
        stable = 0
        prev = self._screen_mirror()
        while frames < SETTLE_TIMEOUT_FRAMES:
            self._tick_impl(SETTLE_POLL_FRAMES)
            frames += SETTLE_POLL_FRAMES
            cur = self._screen_mirror()
            stable = stable + 1 if (reader.is_engine_idle() and cur == prev) else 0
            prev = cur
            needed = SETTLE_STABLE_BLANK if self._bottom_blank(cur) else SETTLE_STABLE_SAMPLES
            if stable >= needed:
                return
        logger.info(f"[Emulator] Settle timeout after {frames} frames (game busy)")

    def get_coordinates(self):
        """
        Returns the player's current coordinates from game memory.
        Returns:
            tuple[int, int]: (x, y) coordinates
        """
        return self._run_on_pyboy(
            lambda: PokemonRedReader(self.pyboy.memory).read_coordinates()
        )

    def get_active_dialog(self):
        """
        Returns the active dialog text from game memory.
        Returns:
            str: Dialog text
        """
        dialog = self._run_on_pyboy(
            lambda: PokemonRedReader(self.pyboy.memory).read_dialog()
        )
        if dialog:
            return dialog
        return None

    def get_location(self):
        """
        Returns the player's current location name from game memory.
        Returns:
            str: Location name
        """
        return self._run_on_pyboy(
            lambda: PokemonRedReader(self.pyboy.memory).read_location()
        )

    def _get_direction(self, array):
        """Determine the player's facing direction from the sprite pattern."""
        # Look through the array for any 2x2 grid containing numbers 0-3
        rows, cols = array.shape

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Extract 2x2 grid
                grid = array[i : i + 2, j : j + 2].flatten()

                # Check for each direction pattern
                if list(grid) == [0, 1, 2, 3]:
                    return "down"
                elif list(grid) == [4, 5, 6, 7]:
                    return "up"
                elif list(grid) == [9, 8, 11, 10]:
                    return "right"
                elif list(grid) == [8, 9, 10, 11]:
                    return "left"

        return "no direction found"

    def _downsample_array(self, arr):
        """Downsample an 18x20 array to 9x10 by averaging 2x2 blocks."""
        # Ensure input array is 18x20
        if arr.shape != (18, 20):
            raise ValueError("Input array must be 18x20")

        # Reshape to group 2x2 blocks and take mean
        return arr.reshape(9, 2, 10, 2).mean(axis=(1, 3))

    def get_collision_map(self):
        """
        Creates a simple ASCII map showing player position, direction, terrain and sprites.
        Returns:
            str: A string representation of the ASCII map with legend
        """
        # Snapshot terrain, movement and sprite data atomically — the
        # game keeps running between submissions to the PyBoy thread
        def snapshot():
            return (
                self.pyboy.game_wrapper.game_area(),
                self.pyboy.game_wrapper.game_area_collision(),
                self.get_sprites(),
            )

        full_map, collision_map, sprite_locations = self._run_on_pyboy(snapshot)
        downsampled_terrain = self._downsample_array(collision_map)

        # Get character direction from the full map
        direction = self._get_direction(full_map)
        if direction == "no direction found":
            return None

        # Direction symbols
        direction_chars = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        player_char = direction_chars.get(direction, "P")

        # Create the ASCII map
        horizontal_border = "+" + "-" * 10 + "+"
        lines = [horizontal_border]

        # Create each row
        for i in range(9):
            row = "|"
            for j in range(10):
                if i == 4 and j == 4:
                    # Player position with direction
                    row += player_char
                elif (j, i) in sprite_locations:
                    # Sprite position
                    row += "S"
                else:
                    # Terrain representation
                    if downsampled_terrain[i][j] == 0:
                        row += "█"  # Wall
                    else:
                        row += "·"  # Path
            row += "|"
            lines.append(row)

        # Add bottom border
        lines.append(horizontal_border)

        # Add legend
        lines.extend(
            [
                "",
                "Legend:",
                "█ - Wall/Obstacle",
                "· - Path/Walkable",
                "S - Sprite",
                f"{direction_chars['up']}/{direction_chars['down']}/{direction_chars['left']}/{direction_chars['right']} - Player (facing direction)",
            ]
        )

        # Join all lines with newlines
        return "\n".join(lines)

    def get_valid_moves(self):
        """
        Returns a list of valid moves (up, down, left, right) based on the collision map.
        Returns:
            list[str]: List of valid movement directions
        """
        # Get collision map
        collision_map = self._run_on_pyboy(
            lambda: self.pyboy.game_wrapper.game_area_collision()
        )
        terrain = self._downsample_array(collision_map)

        # Player is always at position (4,4) in the 9x10 downsampled map
        valid_moves = []

        # Check each direction
        if terrain[3][4] != 0:  # Up
            valid_moves.append("up")
        if terrain[5][4] != 0:  # Down
            valid_moves.append("down")
        if terrain[4][3] != 0:  # Left
            valid_moves.append("left")
        if terrain[4][5] != 0:  # Right
            valid_moves.append("right")

        return valid_moves

    def _can_move_between_tiles(self, tile1: int, tile2: int, tileset: str) -> bool:
        """
        Check if movement between two tiles is allowed based on tile pair collision data.

        Args:
            tile1: The tile being moved from
            tile2: The tile being moved to
            tileset: The current tileset name

        Returns:
            bool: True if movement is allowed, False if blocked
        """
        # Tile pair collision data
        TILE_PAIR_COLLISIONS_LAND = [
            ("CAVERN", 288, 261),
            ("CAVERN", 321, 261),
            ("FOREST", 304, 302),
            ("CAVERN", 298, 261),
            ("CAVERN", 261, 289),
            ("FOREST", 338, 302),
            ("FOREST", 341, 302),
            ("FOREST", 342, 302),
            ("FOREST", 288, 302),
            ("FOREST", 350, 302),
            ("FOREST", 351, 302),
        ]

        TILE_PAIR_COLLISIONS_WATER = [
            ("FOREST", 276, 302),
            ("FOREST", 328, 302),
            ("CAVERN", 276, 261),
        ]

        # Check both land and water collisions
        for ts, t1, t2 in TILE_PAIR_COLLISIONS_LAND + TILE_PAIR_COLLISIONS_WATER:
            if ts == tileset:
                # Check both directions since collisions are bidirectional
                if (tile1 == t1 and tile2 == t2) or (tile1 == t2 and tile2 == t1):
                    return False

        return True

    def get_sprites(self, debug=False):
        """
        Get the location of all of the sprites on the screen.
        returns set of coordinates that are (column, row)
        """
        # Group sprites by their exact Y coordinate
        sprites_by_y = {}

        sprites = self._run_on_pyboy(
            lambda: [self.pyboy.get_sprite(i) for i in range(40)]
        )
        for i, sp in enumerate(sprites):
            if sp.on_screen:
                x = int(sp.x / 160 * 10)
                y = int(sp.y / 144 * 9)
                orig_y = sp.y

                if orig_y not in sprites_by_y:
                    sprites_by_y[orig_y] = []
                sprites_by_y[orig_y].append((x, y, i))

        # Sort Y coordinates
        y_positions = sorted(sprites_by_y.keys())
        bottom_sprite_tiles = set()

        if debug:
            print("\nSprites grouped by original Y:")
            for orig_y in y_positions:
                sprites = sprites_by_y[orig_y]
                print(f"Y={orig_y}:")
                for x, grid_y, i in sprites:
                    print(f"  Sprite {i}: x={x}, grid_y={grid_y}")

        SPRITE_HEIGHT = 8

        # First, group sprites by X coordinate for each Y level
        for i in range(len(y_positions) - 1):
            y1 = y_positions[i]
            y2 = y_positions[i + 1]

            if y2 - y1 == SPRITE_HEIGHT:
                # Group sprites by X coordinate at each Y level
                sprites_at_y1 = {s[0]: s for s in sprites_by_y[y1]}  # x -> sprite info
                sprites_at_y2 = {s[0]: s for s in sprites_by_y[y2]}

                # Only match sprites that share the same X coordinate
                for x in sprites_at_y2:
                    if x in sprites_at_y1:  # If there's a matching top sprite at this X
                        bottom_sprite = sprites_at_y2[x]
                        bottom_sprite_tiles.add((x, bottom_sprite[1]))
                        if debug:
                            print(f"\nMatched sprites at x={x}, Y1={y1}, Y2={y2}")

        return bottom_sprite_tiles

    def find_path(self, target_row: int, target_col: int) -> tuple[str, list[str]]:
        """
        Finds the most efficient path from the player's current position (4,4) to the target position.
        If the target is unreachable, finds path to nearest accessible spot.
        Allows ending on a wall tile if that's the target.
        Takes into account terrain, sprite collisions, and tile pair collisions.

        Args:
            target_row: Row index in the 9x10 downsampled map (0-8)
            target_col: Column index in the 9x10 downsampled map (0-9)

        Returns:
            tuple[str, list[str]]: Status message and sequence of movements
        """
        # Snapshot collision map, terrain, sprites, and tileset atomically
        def snapshot():
            return (
                self.pyboy.game_wrapper.game_area_collision(),
                self.get_sprites(),
                self.pyboy.game_wrapper._get_screen_background_tilemap(),
                PokemonRedReader(self.pyboy.memory).read_tileset(),
            )

        collision_map, sprite_locations, full_map, tileset = self._run_on_pyboy(snapshot)
        terrain = self._downsample_array(collision_map)

        # Start at player position (always 4,4 in the 9x10 grid)
        start = (4, 4)
        end = (target_row, target_col)

        # Validate target position
        if not (0 <= target_row < 9 and 0 <= target_col < 10):
            return "Invalid target coordinates", []

        # A* algorithm
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        # Track closest reachable point
        closest_point = start
        min_distance = heuristic(start, end)

        def reconstruct_path(current):
            path = []
            while current in came_from:
                prev = came_from[current]
                if prev[0] < current[0]:
                    path.append("down")
                elif prev[0] > current[0]:
                    path.append("up")
                elif prev[1] < current[1]:
                    path.append("right")
                else:
                    path.append("left")
                current = prev
            path.reverse()
            return path

        while open_set:
            _, current = heapq.heappop(open_set)

            # Check if we've reached target
            if current == end:
                path = reconstruct_path(current)
                is_wall = terrain[end[0]][end[1]] == 0
                if is_wall:
                    return (
                        f"Partial Success: Your target location is a wall. In case this is intentional, attempting to navigate there.",
                        path,
                    )
                else:
                    return (
                        f"Success: Found path to target at ({target_row}, {target_col}).",
                        path,
                    )

            # Track closest point
            current_distance = heuristic(current, end)
            if current_distance < min_distance:
                closest_point = current
                min_distance = current_distance

            # If we're next to target and target is a wall, we can end here
            if (abs(current[0] - end[0]) + abs(current[1] - end[1])) == 1 and terrain[
                end[0]
            ][end[1]] == 0:
                path = reconstruct_path(current)
                # Add final move onto wall
                if end[0] > current[0]:
                    path.append("down")
                elif end[0] < current[0]:
                    path.append("up")
                elif end[1] > current[1]:
                    path.append("right")
                else:
                    path.append("left")
                return (
                    f"Success: Found path to position adjacent to wall at ({target_row}, {target_col}).",
                    path,
                )

            # Check all four directions
            for dr, dc, direction in [
                (1, 0, "down"),
                (-1, 0, "up"),
                (0, 1, "right"),
                (0, -1, "left"),
            ]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds
                if not (0 <= neighbor[0] < 9 and 0 <= neighbor[1] < 10):
                    continue
                # Skip walls unless it's the final destination
                if terrain[neighbor[0]][neighbor[1]] == 0 and neighbor != end:
                    continue
                # Skip sprites unless it's the final destination
                if (neighbor[1], neighbor[0]) in sprite_locations and neighbor != end:
                    continue

                # Check tile pair collisions
                # Get bottom-left tile of each 2x2 block
                current_tile = full_map[current[0] * 2 + 1][
                    current[1] * 2
                ]  # Bottom-left tile of current block
                neighbor_tile = full_map[neighbor[0] * 2 + 1][
                    neighbor[1] * 2
                ]  # Bottom-left tile of neighbor block
                if not self._can_move_between_tiles(
                    current_tile, neighbor_tile, tileset
                ):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If target unreachable, return path to closest point
        if closest_point != start:
            path = reconstruct_path(closest_point)
            return (
                f"Partial Success: Could not reach the exact target, but found a path to the closest reachable point.",
                path,
            )

        return (
            "Failure: No path is visible to the chosen location. You may need to explore a totally different path to get where you're trying to go.",
            [],
        )

    def get_state_from_memory(self) -> str:
        """
        Reads the game state from memory and returns a string representation of it.
        """
        return self._run_on_pyboy(self._get_state_from_memory_impl)

    def _get_state_from_memory_impl(self) -> str:
        reader = PokemonRedReader(self.pyboy.memory)
        memory_str = ""

        name = reader.read_player_name()
        if name == "NINTEN":
            name = "Not yet set"
        rival_name = reader.read_rival_name()
        if rival_name == "SONY":
            rival_name = "Not yet set"

        # Get valid moves
        valid_moves = self.get_valid_moves()
        valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"

        memory_str += f"Player: {name}\n"
        memory_str += f"Rival: {rival_name}\n"
        memory_str += f"Money: ${reader.read_money()}\n"
        memory_str += f"Location: {reader.read_location()}\n"
        memory_str += f"Coordinates: {reader.read_coordinates()}\n"
        memory_str += f"Valid Moves: {valid_moves_str}\n"
        memory_str += f"Badges: {', '.join(reader.read_badges())}\n"

        # Inventory
        memory_str += "Inventory:\n"
        for item, qty in reader.read_items():
            memory_str += f"  {item} x{qty}\n"

        # Dialog
        dialog = reader.read_dialog()
        if dialog:
            memory_str += f"Dialog: {dialog}\n"
        else:
            memory_str += "Dialog: None\n"

        # Party Pokemon
        memory_str += "\nPokemon Party:\n"
        for pokemon in reader.read_party_pokemon():
            memory_str += f"\n{pokemon.nickname} ({pokemon.species_name}):\n"
            memory_str += f"Level {pokemon.level} - HP: {pokemon.current_hp}/{pokemon.max_hp}\n"
            memory_str += f"Types: {pokemon.type1.name}{', ' + pokemon.type2.name if pokemon.type2 else ''}\n"
            for move, pp in zip(pokemon.moves, pokemon.move_pp, strict=True):
                memory_str += f"- {move} (PP: {pp})\n"
            if pokemon.status != StatusCondition.NONE:
                memory_str += f"Status: {pokemon.status.get_status_name()}\n"

        return memory_str

    def stop(self):
        self._stop_keepalive.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=2.0)
        try:
            self._run_on_pyboy(self.pyboy.stop)
        finally:
            self._executor.shutdown(wait=False)