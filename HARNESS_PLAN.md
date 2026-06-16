# Harness Improvement Plan

Ideas collected from other "LLM plays Pokemon" projects, mapped onto this codebase.
Main sources: [The Making of Gemini Plays Pokemon](https://blog.jcz.dev/the-making-of-gemini-plays-pokemon),
[Claude vs Gemini harness comparison](https://www.lesswrong.com/posts/7mqp8uRnnPdbBzJZE/is-gemini-now-better-than-claude-at-pokemon),
[cicero225/llm_pokemon_scaffold](https://github.com/cicero225/llm_pokemon_scaffold) (evolved from the same
Hershey starter we forked), [Continual Harness paper](https://arxiv.org/html/2605.09998),
[PokéLLMon](https://arxiv.org/abs/2402.01118), [NousResearch/pokemon-agent](https://github.com/NousResearch/pokemon-agent),
[PWhiddy RL experiments](https://github.com/PWhiddy/PokemonRedExperiments).

**Guiding constraint:** the Continual Harness paper measured that rich scaffolds *hurt* cheap models
(Gemini Flash-Lite scored worse with a fancy harness than a minimal one; gains only appeared on
Pro-tier models). We run Gemini 3 Flash, so this plan favors cheap deterministic code over
sub-agents and LLM-side complexity. Every successful project also reported *removing* scaffolding
as models improved — build the minimum that fixes an observed failure.

---

## Tier 1 — Spatial memory (the proven biggest win) ✅ IMPLEMENTED

These three items are one coherent feature. The LessWrong comparison isolated them as the reason
Gemini got ~2x further than Claude using ~1/3 the actions.

> Implemented in `src/pokemon_agent/emulator/world_map.py` + emulator/tool/prompt wiring.
> Deviations from the plan below: the screen-grid collision map was *replaced* by the explored-map
> render (one coordinate language everywhere, no fallback screen mode), door/warp coordinates are
> read from the map header in RAM (`0xD3AE`) and listed alongside the frontier, and settle
> detection now also watches the palette registers — warp fades are palette-only and previously
> let observations read torn mid-warp state.

### 1. Fog-of-war map: remember every tile ever seen

**Problem:** the model has no spatial memory. Each turn it only sees the current screen, so it
re-explores rooms, forgets where doors were, and walks in circles for hours.

**Proposed solution:**
- New module `src/pokemon_agent/emulator/world_map.py` holding a per-map dict:
  `{map_id: {(global_x, global_y): tile_info}}` where `tile_info` is walkable/wall/NPC/unknown.
- Update it after every settle inside `Emulator.press_buttons`: convert the current 9x10 screen
  collision grid to global map coordinates (player sits at screen cell (4, 4);
  `MemoryReader.read_coordinates()` gives the player's global position, so
  `global = player_pos + (screen_cell - (4, 4))`).
- Render it as a text grid (`#` wall, `.` walkable, `?` unexplored edge, `P` player) in
  `observe_after_action` for the **current map only**. GPP found text maps beat rendered images.
- Persist it next to the save state (e.g. `states/game-NNN.map.json`) so resuming a run resumes
  the map too — mirrors how cicero225 pickles scaffold state alongside `save.state`.

### 2. Frontier list: tell the model where it hasn't looked

**Problem:** the model claims "I'm stuck / softlocked" while unexplored exits exist. GPP's author
calls the frontier list his single most effective anti-stuck device.

**Proposed solution:**
- BFS over the fog-of-war map from the player's position; collect tiles that are *reachable* and
  border *unexplored* space.
- Append to every observation: `Unexplored reachable spots on this map: (12,4), (3,9), ...`
  (capped, nearest first, with BFS distance).
- One line in `prompts/system/game_player.j2`: "if you feel stuck, go to an unexplored spot first."
- Plain code in `world_map.py`, no LLM cost.

### 3. Off-screen navigation: travel to any remembered place

**Problem:** `navigate_to` ([navigate_to.py](src/pokemon_agent/agent/tools/navigate_to.py)) only
paths to the visible 9x10 screen grid. The model can see a staircase, walk away, and have no way
back except step-by-step button presses.

**Proposed solution:**
- Extend `navigate_to` to accept global map coordinates (the same coordinate system shown on the
  fog-of-war map and the frontier list — one coordinate language everywhere).
- A* over the fog-of-war map instead of the on-screen grid; execute the path with the existing
  per-press settle loop, re-planning if a tile turns out blocked (NPCs move).
- Keep the current screen-grid mode as a fallback when the target is visible.
- Caution from cicero225: don't claim walkability for tiles adjacent to sprites or at warp/map
  edges — collision data lies there. Mark those `?`.

---

## Tier 2 — Ground truth from RAM (kills hallucinations)

### 4. Event-flag diff: the game's own progress tracker

**Problem:** the model misremembers what it has accomplished ("did I already beat Mt. Moon?").
Claude once deleted its notes and concluded it had beaten a cave it never entered.

**Proposed solution:**
- Pokemon Red stores quest progress as a bitfield at `0xD747–0xD87E` (~2,300 event flags).
- Snapshot it before each tool action; after settle, diff. On change, append to the tool result:
  `Game progress: a new event flag was just set — something you did officially counted.`
- Optionally map known flag indices to names later (pret/pokered's `events.json` via
  [pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer)).
- Bonus for streaming: a flag flip is the perfect trigger for an excited TTS line.
- ~30 lines in [memory_reader.py](src/pokemon_agent/emulator/memory_reader.py) + the tools.

### 5. RAM ground-truth audit: never let the model track numbers mentally

**Problem:** anything the model tracks across turns drifts. GPP's Gemini was convinced its moves
had 0 PP for hours; no prompt fixed it — injecting live PP from RAM fixed it instantly.

**Proposed solution:**
- Audit what `observe_after_action` already sends (location, coords, party, badges, money, items,
  dialog) against the full hallucination-prone list: **per-move PP** (top offender), HP/status per
  party member, repel steps, current box space.
- Verify `read_party_pokemon` exposes move PP and that it lands in the observation text; add what's
  missing.
- Best single Red RAM crib sheet: [NousResearch red.py](https://github.com/NousResearch/pokemon-agent)
  — includes the gotcha that Gen 1 stores internal species indices, not dex numbers.

### 6. Loop detection (and optional save-state rewind)

**Problem:** the dominant failure mode in every project: repeating the same actions without
progress. The model rarely notices on its own.

**Proposed solution:**
- Track a counter of visits per `(map_id, x, y)` since the last *new* tile or event flag
  (PWhiddy's RL env used heavy revisits as its stuck signal).
- Threshold 1 (soft): inject a system note — "you have revisited the same few tiles N times and
  made no progress; your current approach is not working, consult the unexplored-spots list."
- Threshold 2 (hard, optional later): keep a small deque of automatic PyBoy save states (~130KB
  each, [papercomputeco/pokemon](https://github.com/papercomputeco/pokemon) keeps 8) and offer a
  rewind. Start with the soft nudge only.
- Counters reset on new-tile discovery or event-flag change — both already computed by items 1 & 4.

---

## Tier 3 — Memory & goals across summarization

### 7. Persistent goal slots

**Problem:** after summarization the model wavers on what it was doing; goals dissolve into the
summary prose. GPP added explicit goal tiers because of exactly this.

**Proposed solution:**
- Add `goals: dict` to `GameAgentState` ([state.py](src/pokemon_agent/agent/state.py)):
  `{primary, secondary, tertiary}`.
- New small tool `set_goal(slot, text)` so the model updates them deliberately.
- Render the slots verbatim (never summarized) in every observation and in the
  summary-handoff message ([summary_handoff.j2](prompts/human/summary_handoff.j2)).
- Guard from GPP's "TEA incident" (a hallucinated goal persisted for hours): the prompt should
  tell the model to drop a goal it can't ground in the checkpoint log or current state.

### 8. Append-only checkpoint log

**Problem:** same hallucination class as item 4, but for things the game doesn't flag (e.g.
"reached Cerulean", "taught Cut to Charmander").

**Proposed solution:**
- New tool `mark_checkpoint(achievement)` appending to a list in `GameAgentState`; entries are
  never edited or deleted (append-only is the anti-hallucination property).
- The last ~15 entries render in every observation and survive summarization verbatim.
- cicero225 also resets the loop-detection counter (item 6) on checkpoint — do the same.
- Optionally auto-append event-flag changes (item 4) so the log fills itself.

### 9. Trust-tier summarization

**Problem:** summaries blend RAM facts, model notes, and vision guesses with equal confidence;
one wrong "fact" then poisons everything downstream. Every project reports the model's own notes
as the top failure surface (a single wrong note cost GPP ~100 hours).

**Proposed solution:**
- Rewrite [summarize_request.j2](prompts/human/summarize_request.j2) with an explicit trust
  hierarchy (cicero225's wording works well):
  1. RAM-derived data (location, party, badges, events) — absolute, never wrong
  2. Checkpoint log
  3. Fog-of-war map
  4. The model's own earlier statements/notes
  5. Claims derived from looking at screenshots — least trustworthy
- Instruct the summarizer to **delete** coordinates and map-layout claims that came only from
  vision, and to flag (not assert) anything from tiers 4–5 that contradicts tiers 1–3.
- Prompt-only change, zero code.

---

## Tier 4 — Battle smarts (when battles become the bottleneck)

### 10. Inject the type chart and battle feedback

**Problem:** models misremember type matchups (GPT-4 is only ~84% accurate on the 18x18 chart) and
plan battles blind to PP.

**Proposed solution (in order of measured impact, all from PokéLLMon):**
- When `in battle` (battle type byte `0xD057` != 0), append to the observation: both active
  Pokemon's types + a one-line effectiveness summary ("Water is super effective vs your opponent;
  their Electric moves are super effective vs you"). Measured +19 points win rate alone.
- Turn feedback: HP deltas since last turn and the last move's effectiveness text. +10 points.
- Counterintuitive finding: long chain-of-thought *hurt* battles (panic-switching). Keep battle
  reasoning short; don't add a battle sub-agent.
- If battle-specific tools ever get added, expose **item use** first — its removal crushed win
  rate (81% → 33%) far more than removing switching.

---

## Tier 5 — Streaming phase (existing roadmap, informed by research)

### 11. Help channel / Twitch

- Working reference for chat-into-prompt:
  [nichosta/GeminiPlaysPokemonLive](https://github.com/nichosta/GeminiPlaysPokemonLive)
  (Twitch chat lines injected into the decision prompt; a text file stands in for chat offline —
  matching our planned ChatSource abstraction).
- NousResearch tags narration events by type (THINK / DECIDE / ACT / MILESTONE / ALERT); useful
  later so TTS can prioritize milestone moments (event-flag flips from item 4) over routine
  chatter.

---

## Explicitly not doing (for now)

- **Sub-agents** (pathfinder / critic / boulder-strategist personas): the big wins reported for
  these were on Pro-tier models; on Flash-class models rich scaffolds measured *worse* than
  minimal ones. Items 1–3 replace the pathfinder with plain A*.
- **World knowledge graph:** tried and removed by GPP — the model wasted turns on bookkeeping.
- **Free-form notepad tool:** highest-risk memory surface in every project (one bad note = days
  lost). Goal slots + append-only checkpoints cover the need with less poisoning surface.
- **Screenshot coordinate/label overlays:** our RAM collision map + text fog-of-war already covers
  this cheaper; revisit only if the model demonstrably misreads scenes.
- **LLM-written skills / code execution:** Voyager-style; powerful but heavy, and model-capability
  gated.

## Suggested order

1. Items **1 + 2 + 3** together (one feature: world map, frontier, long-range navigate).
2. Item **4** (event-flag diff) — tiny, immediately useful, feeds items 6 and 8.
3. Items **7 + 8 + 9** (goals, checkpoints, trust-tier summarization) — hardens long runs.
4. Item **5** (RAM audit) alongside, opportunistically.
5. Item **6** soft nudge; hard rewind only if loops persist.
6. Tier 4 when battles become the bottleneck; Tier 5 with the help-channel phase.
