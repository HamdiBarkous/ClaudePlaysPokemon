import concurrent.futures
import logging
import os
import threading

from pokemon_agent.emulator.memory_reader import PokemonRedReader, StatusCondition
from pokemon_agent.emulator.world_map import PLAYER_COL, PLAYER_ROW, WorldMap
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

NAV_MAX_PRESSES = 100         # ceiling on a single navigate_to walk


class Emulator:
    """PyBoy wrapper where a single dedicated thread owns the emulator.

    SDL is not thread-safe: when ticks (which render the window) come from
    changing threads — LangGraph runs tools in worker threads — the window
    silently stops updating. So PyBoy is constructed and driven exclusively on
    one thread; every other thread submits work to it via _run_on_pyboy.
    """

    def __init__(self, rom_path, headless=True, sound=False):
        self._headless = headless
        self.world_map = WorldMap()
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

                prev_map = PokemonRedReader(self.pyboy.memory).read_map_id()
                self.pyboy.button_press(button)
                self._tick_impl(10)   # Press briefly
                self.pyboy.button_release(button)
                self._settle_impl()   # Run until the game stops reacting
                if PokemonRedReader(self.pyboy.memory).read_map_id() != prev_map:
                    self._wait_map_ready_impl()
                self._update_world_map_impl()

                keyframes.append(Image.fromarray(self.pyboy.screen.ndarray.copy()))
                results.append(f"Pressed {button}")
            return results, keyframes

        results, keyframes = self._run_on_pyboy(impl)
        return "\n".join(results), keyframes

    def _screen_mirror(self) -> bytes:
        """wTileMap with the blinking continue-arrow masked out, plus the
        palette registers (BGP/OBP0/OBP1).

        Palettes are how screen fades happen — the tilemap freezes during a
        warp fade while the map is half-loaded, so without them the fade
        looks settled and observations read torn state (new map number, old
        coordinates).
        """
        reader = PokemonRedReader(self.pyboy.memory)
        tilemap = reader.read_tilemap_buffer().replace(bytes([CONTINUE_ARROW_TILE]), b"\x00")
        palettes = bytes(
            [self.pyboy.memory[0xFF47], self.pyboy.memory[0xFF48], self.pyboy.memory[0xFF49]]
        )
        return tilemap + palettes

    @staticmethod
    def _bottom_blank(mirror: bytes) -> bool:
        """Whether the textbox area (bottom 6 tile rows) is blank."""
        bottom = mirror[12 * 20 : 18 * 20]
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
            # is_player_placed: the warp fade freezes the screen (stable!)
            # while the new map is half-loaded — don't settle inside it
            quiet = (
                reader.is_engine_idle()
                and reader.is_player_placed()
                and cur == prev
            )
            stable = stable + 1 if quiet else 0
            prev = cur
            needed = SETTLE_STABLE_BLANK if self._bottom_blank(cur) else SETTLE_STABLE_SAMPLES
            if stable >= needed:
                return
        logger.info(f"[Emulator] Settle timeout after {frames} frames (game busy)")

    def _wait_map_ready_impl(self) -> None:
        """After a map change, run until the warp transition fully completes.

        The fade can outlive the settle (a frozen screen looks settled while
        the new map is half-loaded: map number updated, player coordinates
        and map header still the old map's). Joypad input is disabled for
        the whole transition, so run until it's re-enabled and the player is
        placed — or a dialog opens, meaning a script took over (scripted map
        entries like first entering Oak's Lab).
        """
        reader = PokemonRedReader(self.pyboy.memory)
        frames = 0
        while frames < SETTLE_TIMEOUT_FRAMES:
            if reader.read_dialog():
                return
            if reader.read_joy_ignore() == 0 and reader.is_player_placed():
                return
            self._tick_impl(SETTLE_POLL_FRAMES)
            frames += SETTLE_POLL_FRAMES
        logger.info("[Emulator] Map transition still busy after wait ceiling")

    def _update_world_map_impl(self) -> None:
        """Record the visible screen into the fog-of-war map (PyBoy thread).

        Skipped whenever the screen isn't a clean overworld view: text boxes
        and menus overwrite the tile data they cover, a mid-animation engine
        means the coordinates may not match the screen yet, and an ignored
        joypad means a warp or script is mid-flight.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        if not reader.is_engine_idle() or not reader.is_player_placed():
            return
        if reader.read_joy_ignore():
            return
        if reader.read_dialog():
            return
        if self._get_direction(self.pyboy.game_wrapper.game_area()) == "no direction found":
            return
        terrain = self._downsample_array(self.pyboy.game_wrapper.game_area_collision())
        self.world_map.update_from_screen(
            reader.read_map_id(),
            reader.read_coordinates(),
            terrain,
            self.get_sprites(),
        )

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

    def get_world_map_text(self):
        """Text view of the explored map around the player, or None outside
        the overworld (battles/menus, where map coordinates mean nothing).

        Includes the fog-of-war terrain with coordinate rulers, current
        on-screen sprites, the map's door/warp coordinates (RAM ground
        truth), and the reachable-unexplored frontier list.
        """
        # Snapshot everything atomically — the game keeps running between
        # submissions to the PyBoy thread
        def snapshot():
            self._update_world_map_impl()
            reader = PokemonRedReader(self.pyboy.memory)
            return (
                self.pyboy.game_wrapper.game_area(),
                reader.read_map_id(),
                reader.read_location(),
                reader.read_coordinates(),
                reader.read_warps(),
                self.get_sprites(),
            )

        area, map_id, location, pos, warps, sprites = self._run_on_pyboy(snapshot)
        direction = self._get_direction(area)
        if direction == "no direction found":
            return None

        px, py = pos
        sprites_global = {
            (px + col - PLAYER_COL, py + row - PLAYER_ROW)
            for col, row in sprites
            if (col, row) != (PLAYER_COL, PLAYER_ROW)
        }
        return self.world_map.render(
            map_id, location, pos, direction, warps, sprites_global
        )

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

    def navigate_to_global(self, x: int, y: int) -> tuple[str, list]:
        """Walk to global map coordinate (x, y) over explored terrain.

        Plans over the fog-of-war world map, then walks one step at a time,
        verifying each move against RAM and re-planning around surprises the
        collision data can't see (moving NPCs, ledges, spin tiles). Stops
        early when a warp triggers or something appears on screen.

        Returns:
            tuple[str, list[Image.Image]]: Result text and per-step keyframes.
        """
        def read_state():
            reader = PokemonRedReader(self.pyboy.memory)
            return (
                reader.read_map_id(),
                reader.read_coordinates(),
                bool(reader.read_dialog()),
            )

        def arrived_message():
            warps = self._run_on_pyboy(
                lambda: PokemonRedReader(self.pyboy.memory).read_warps()
            )
            if (x, y) in warps:
                # Exit mats inside buildings only trigger when you step off
                # them toward the map edge
                return (
                    f"Arrived at ({x}, {y}) — standing on the door/warp tile. "
                    "If nothing happened, step once more toward the exit "
                    "(usually down) to go through."
                )
            return f"Arrived at ({x}, {y})."

        start_map, pos, dialog = self._run_on_pyboy(read_state)
        if dialog:
            return (
                "Can't walk right now — there's a dialog, menu, or battle on "
                "screen. Deal with it first (press_buttons).",
                [],
            )
        keyframes = []
        blocked: set = set()
        for _ in range(NAV_MAX_PRESSES):
            if pos == (x, y):
                return arrived_message(), keyframes
            path, bump = self.world_map.find_path(start_map, pos, (x, y), blocked)
            if not path:
                where = f"stopped at {pos}" if keyframes else f"you are at {pos}"
                return (
                    f"No walkable route to ({x}, {y}) through explored terrain — "
                    f"{where}. Explore toward it first, or pick a reachable target.",
                    keyframes,
                )
            step = path[0]
            _, kfs = self.press_buttons([step])
            keyframes.extend(kfs)
            new_map, new_pos, dialog = self._run_on_pyboy(read_state)
            if new_map != start_map:
                return (
                    f"Walked through a door/warp at {pos} into a new area; "
                    "navigation stopped there.",
                    keyframes,
                )
            if dialog:
                return (
                    f"Navigation interrupted at {new_pos} — something came up "
                    "on screen (dialog or battle).",
                    keyframes,
                )
            if new_pos == pos:
                if bump and len(path) == 1:
                    return (
                        f"Standing next to ({x}, {y}) and facing it — the tile "
                        "itself is blocked (a person or obstacle).",
                        keyframes,
                    )
                blocked.add((pos, step))
            pos = new_pos
        return (
            f"Stopped after {NAV_MAX_PRESSES} steps without reaching ({x}, {y}); "
            f"currently at {pos}.",
            keyframes,
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