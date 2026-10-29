# ARC-AGI-3 Active Causal Agent: Complete Project Handoff for AI-Assisted Development
## Purpose of this document
This document is a self-contained technical handoff for humans and AI assistants continuing an ARC-AGI-3 course project. It reconstructs the project’s motivation, design decisions, experimental history, present codebase, observed successes and failures, neural-network roadmap, and recommended next steps.

It is intentionally explicit about the difference between **implemented components**, **partially implemented ideas**, and **future proposals**. An AI receiving this document should not assume that a component exists merely because it appears in the roadmap.

The working artifact discussed here is the current `agent.py`, whose main class is `LS20ExtendedStateAgent`. It is an observation-only, causal, symbolic agent specialized to the visual grammar of the public `ls20` environment. It does not inspect hidden environment state or game source while acting.[^1]
## Project context
ARC-AGI-3 is an interactive reasoning benchmark in which an agent must explore unfamiliar environments, infer goals, build adaptable world models, and learn continuously. The official agent framework exposes game observations and legal actions, and the official Kaggle starter is designed around editing an agent file, testing locally, and building a submission from the same implementation.[^2][^3][^4][^5]

Unlike static ARC tasks, the agent cannot infer everything from a single input-output example. It must choose actions, observe consequences, distinguish causal effects from incidental visual changes, preserve useful knowledge, and plan efficiently.

The project began from one key human observation about `ls20`: people do not discover the game by pressing controls uniformly at random. They identify visually unusual locations, deliberately travel to them, observe what changes, and use those interventions to infer the game’s rules. For example, a special tile may rotate a carried geometric symbol, while a door displays the target symbol required for progression.

The central project hypothesis is therefore:

> **An agent that detects salient interactive entities, performs targeted causal experiments, stores compact transition models, and plans with those learned models will solve unfamiliar interactive tasks more efficiently than random exploration or a purely reactive neural policy.**
## One-sentence architecture
The proposed final system is a **neuro-symbolic active causal agent**: neural perception converts pixels into object-centric state, a reduced-order latent world model tracks task-relevant dynamics, active experimentation identifies causal operators, and symbolic or model-based planning composes those operators while verifying every action against reality.
## Core reasoning loop
The intended agent loop is:

1. Observe the current frame and recent temporal context.
2. Parse the scene into a compact semantic state.
3. Detect the controllable entity, traversable space, goals, resources, special entities, and dynamic mechanisms.
4. Maintain multiple hypotheses about what unknown entities do.
5. Select an experiment that trades off expected information gain, progress, path cost, and risk.
6. Navigate to the selected interaction point.
7. Compare the predicted and observed consequences.
8. Update the causal model and confidence in each hypothesis.
9. Search for a plan to a goal using the learned transition model.
10. Execute one action, validate the prediction, and replan if the observation disagrees.

A conceptual experiment-selection utility is

\[
U(o)=\alpha\,\mathbb{E}[\Delta H(o)]
+\beta\,P(\text{progress}\mid o)
-\gamma\,C_{\text{path}}(o)
-\eta\,R(o),
\]

where expected entropy reduction rewards informative interventions, progress probability rewards task advancement, path cost discourages wasteful travel, and risk discourages experiments likely to consume a life or exhaust a resource.

The current code implements a simplified, mostly deterministic approximation to this loop rather than the full Bayesian formulation.[^1]
## Development history
### Initial idea: random actions
The first baseline under consideration was random or nearly random control exploration. This is easy to implement and useful only as a minimum reference. It wastes actions because most random movements do not reach causally informative locations.
### Salient-tile exploration
The first meaningful prototype detected visually unusual cells, navigated to the nearest candidate using breadth-first search, compared frames before and after contact, and updated a small hypothesis vector. The initial hypothesis set included:

- Rotation or shape transformation.
- Recoloring.
- Resource refill.
- Goal or exit.
- Teleportation or nonlocal motion.
- Hazard.
- No effect.

The associated README correctly described this as a research baseline and data collector, not a universal ARC-AGI-3 submission. It also proposed eventually replacing handcrafted perception and effect likelihoods with learned object-centric models.[^6]
### Semantic key model
Frame differences alone could not distinguish movement of the player from changes to the carried symbol. The next version extracted the carried symbol and door targets as 3-by-3 discrete arrays. This enabled explicit classifications:

- `rotate_cw`.
- `rotate_ccw`.
- `rotate_180`.
- `recolor`.
- `reshape`.
- `resource`.

Instead of remembering only that a tile “changes something,” the agent began storing exact mappings from a key state before contact to a key state after contact.
### Joint planning
The planner was extended from position-only BFS to search over a semantic state containing player position, key configuration, remaining resource, consumed pickups, and a coarse dynamic configuration. This allowed planning through transformations and refills rather than merely routing around walls.
### Dynamic transitions
Later levels contained nonlocal displacements and visual mechanisms that invalidated a static grid model. The agent therefore started learning transitions of the form

\[
(p_t,a_t,d_t)\rightarrow(p_{t+1},d_{t+1}),
\]

where `d_t` is a tuple describing detected gates and possible moving blocks.
### Current outcome
The agent has repeatedly solved the first four levels of `ls20`, although action efficiency has varied between revisions. It still fails at level 5. The strongest observed run completed four of seven levels; one run used 13, 45, 49, and 61 actions on levels 1–4, while later experimental changes regressed level 4 and did not solve level 5. These numbers come from the supplied execution logs and should be treated as experiment history, not benchmark-wide performance.

The level-5 failure is now the most useful diagnostic case because it combines transformations, resources, moving or periodic mechanisms, nonlocal motion, and planning under uncertainty.
## Current code status
### Important truth
**The current `agent.py` contains no trained neural network.** It uses NumPy, connected-component analysis, fixed color and geometry rules, counters, dictionaries, hashes, and breadth-first search. Neural networks, ROM learning, and object-centric latent dynamics are planned extensions, not current implementation.[^1]

This distinction must remain explicit in presentations and future AI conversations.
### Runtime interface
The current script creates an `arc_agi.Arcade`, loads a game with frame data enabled, resets it, and repeatedly calls `next_action`. Each selected action is submitted with a short reasoning label. The official starter uses a different wrapper contract in which a `MyAgent` class implements `is_done` and `choose_action`; adapting the research class to that submission contract should be a thin interface layer rather than a redesign of the core algorithms.[^1][^4]
### Main class
The implemented agent is:

```python
class LS20ExtendedStateAgent:
    ...
```

Its documented planning state is:

```text
(player, key, resource, consumed pickups, dynamic configuration)
```

Its persistent cross-level knowledge consists primarily of tile-effect models indexed by visual signature. Position-specific dynamic transitions are reset on each new level.[^1]
## Current data model
### Type aliases
```python
Pos = tuple[int, int]
Key = tuple[int, ...]
DynamicConfig = tuple[tuple[str, int, int, int, int], ...]
```

`Key` is a flattened 3-by-3 visual symbol. Background cells use color value 5. `DynamicConfig` is a sorted tuple of gate or block descriptors.
### `TileModel`
`TileModel` stores learned behavior for one visual tile signature:

```python
@dataclass
class TileModel:
    signature: str
    votes: Counter
    transitions: dict[Key, Key]
    color_map: dict[int, int]
    shape_map: dict[Mask, Mask]
    visits: int
```

The dominant role is selected by majority vote. Exact observed key transitions take precedence. If no exact transition is available, `predict` applies a role-specific model:

- Matrix rotation for rotation roles.
- Dominant-color substitution for recoloring.
- Binary occupancy-mask mapping for reshaping.

This is a small program-like transition model rather than a neural predictor.[^1]
### `Tile`
A `Tile` contains grid position, visual signature, kind, and visit count. Tile kinds are presently coarse: ordinary mechanisms and pickup-like visuals.
### `Door`
A `Door` associates a grid position with the 3-by-3 target key required for entry.
### `Gate`
A `Gate` stores a visual signature, orientation, image bounding box, and the two adjacent grid cells separated by the detected line object.
### `DynamicTransition`
A dynamic transition records:

```text
source position
selected action
observed destination
associated gate signature, if any
configuration before
configuration after
number of observations
```

The current implementation can remember nonlocal motion conditioned on an exact dynamic configuration, but it does not yet learn a robust periodic object model.[^1]
### `Scene`
`Scene` is the parsed observation:

```text
frame stack
latest frame
level index
player position
inferred grid origin
free cells
visible special tiles
doors and targets
current key
resource amount
lives
gates
dynamic configuration
```
### `Metrics`
The script records game ID, episode, win state, final state, levels completed, action count, interaction count, learned dynamic transitions, semantic repetitions, and learned tile roles.
## Current perception pipeline
### Frame normalization
`frame_stack` converts the environment observation into a NumPy array and selects a consistent frame-stack representation.
### Connected components
`components` performs four-connected component extraction on a Boolean image mask. This primitive supports player detection, symbol extraction, and gate or block detection.
### Player detection
`detect_player` searches for connected components of color 12 satisfying hardcoded size and position constraints. It infers a 5-pixel grid origin from the detected component and converts pixel coordinates to a coarse cell coordinate.[^1]

This is effective for `ls20` but is strongly game-specific.
### Symbol extraction
`extract_symbols` searches for sufficiently large background-color components. It identifies:

- The carried key in a fixed lower HUD region and downsamples a 6-by-6 patch to 3-by-3.
- Door goal symbols as 3-by-3 patches inside 7-by-7 boxes.

The current and target symbols are therefore available as exact discrete vectors rather than unstructured pixels.[^1]
### Resource and lives
Resource and lives are computed from fixed pixel slices in the bottom HUD using known colors. This is another `ls20`-specific assumption.
### Special tiles
The parser scans 5-by-5 macro-cells, identifies floor-like cells, and computes salience from a fixed set of colors. Salient cells become `Tile` objects with SHA-1-based local visual signatures. Player-colored pixels are normalized out of a signature to reduce contamination.
### Doors
Door positions are inferred by mapping goal-symbol image boxes onto the same coarse grid.
### Gates and blocks
`detect_gates` searches for thin color-1 connected components on cell boundaries. `dynamic_configuration` combines those gate boxes with sufficiently large color-4 components considered possible moving blocks.[^1]
### Main perception limitation
The parser is not object-general. It assumes:

- A particular player color and shape.
- A 5-by-5 spatial grid.
- A fixed 3-by-3 key representation.
- Fixed HUD coordinates.
- Specific colors for floor, pickups, gates, and blocks.

These assumptions are acceptable for a domain-specific baseline but prevent direct transfer to visually different games.
## Causal learning
### Before-and-after intervention
After each action, `observe_transition` compares the previous and current `Scene`. It estimates step cost from resource decrease, checks for life resets, learns unexpected motion, identifies the contacted tile, and updates the corresponding effect model.
### Key transition classification
`classify_key_transition` compares two 3-by-3 arrays:

1. Exact equality implies `none`.
2. Equality after a 90-degree clockwise rotation implies `rotate_cw`.
3. Equality after a 90-degree counterclockwise rotation implies `rotate_ccw`.
4. Equality after 180 degrees implies `rotate_180`.
5. Equal occupancy masks with changed values imply `recolor`.
6. Otherwise the effect is `reshape`.
### Generalization inside `ls20`
For recoloring, the agent learns a mapping between dominant colors. For reshaping, it learns a binary occupancy-mask transition and augments that mapping with rotated versions. This rotational augmentation is a useful inductive bias: it allows a learned shape operator to apply to rotated versions of the same shape.[^1]
### Cross-level persistence
`self.models` persists across levels, allowing the meaning of a visual mechanism to transfer. Level-specific positions, consumed resources, blocked edges, and dynamic transitions reset when entering a new level.
### Hypothesis representation
The early prototype described a Bayesian-style hypothesis vector. The current implementation simplifies that idea to categorical votes and deterministic transition dictionaries. It does not calculate posterior probabilities or entropy explicitly.[^1][^6]
## Planning
### Joint state
`joint_plan` uses BFS over

\[
s=(p,k,r,m,d),
\]

where:

- `p` is player position.
- `k` is the current 3-by-3 key.
- `r` is the remaining resource.
- `m` is a bit mask of consumed pickups.
- `d` is the detected dynamic configuration.
### Spatial transitions
For ordinary actions, the planner predicts movement to an adjacent cell. If an exact learned dynamic edge matches the source, action, and configuration, it substitutes the learned destination and next configuration. Known blocked edges are disallowed.
### Semantic transitions
When the next cell contains a known tile, `apply_tile` predicts its effect on the key or resource. Unknown mechanisms are not traversed during ordinary solution planning, although they may be deliberately targeted for experimentation.
### Goal condition
A door can only be entered if the current key exactly equals that door’s target. If a valid route reaches such a door, BFS returns the action sequence.
### Hierarchical subgoal selection
`required_role` decomposes key mismatch into an ordered semantic agenda:

1. If key equals goal, seek the door.
2. If canonical shapes differ, seek `reshape`.
3. Else if dominant colors differ, seek `recolor`.
4. Else seek `rotate`.

This hierarchy reduces search complexity but encodes an assumption that shape, color, and orientation can be solved sequentially without harmful coupling.
### Active experiments
`semantic_experiment` seeks a tile whose learned role matches the currently required operation but whose transition for the present key is unknown. If resources appear insufficient, it first seeks a refill.
### Exploration priority
`choose_plan` attempts, in order:

1. A full solution from the current learned model.
2. A semantic experiment relevant to the current key mismatch.
3. Discovery of unknown or context-unknown tiles.
4. Testing unknown gate crossings.
5. Reaching unexplored semantic states.
6. A random legal fallback.

This ordering intentionally puts causal key experiments ahead of generic dynamic exploration.[^1]
### Closed-loop execution
Although `joint_plan` returns a sequence, `next_action` executes only the first action and reparses the next observation before choosing again. This is a form of receding-horizon or model-predictive control: the plan is continuously reconsidered after real feedback.
## Reduced-order-model interpretation
The project uses the term ROM in a conceptual and architectural sense. The raw observation is a 64-by-64 categorical image, but the planner operates on a much smaller task-relevant state:

\[
z_t=(p_t,k_t,r_t,m_t,d_t).
\]

This is a handcrafted reduced-order model because it discards most pixels while retaining variables believed sufficient for prediction and control.

The analogy with fluid-mechanics ROMs is useful:

| Computational mechanics | ARC agent |
|---|---|
| High-dimensional field | 64-by-64 frame |
| Reduced coordinates | Object and mechanism state |
| Evolution operator | Learned action-conditioned transition |
| Control input | Game action |
| Observation update | New frame after action |
| Model discrepancy | Predicted versus observed semantic state |
| Receding-horizon control | Execute one action, observe, replan |

However, this is **not currently a neural ROM** and does not use POD. A future learned ROM could use an encoder

\[
z_t=E_\theta(I_{t-k:t})
\]

and a latent dynamics model

\[
\hat z_{t+1}=F_\phi(z_t,a_t),
\]

with a decoder or object heads sufficient to predict task-relevant state changes.

A strong design principle is that the latent state should support interventions and planning, not merely image reconstruction.
## Neural-network roadmap
### Current versus proposed
| Component | Current implementation | Proposed neural implementation |
|---|---|---|
| Player detection | Fixed color and geometry | CNN/U-Net heatmap or object detector |
| Traversability | Color counts in 5-by-5 cells | Segmentation network |
| Salient objects | Fixed color set and heuristics | Objectness/affordance network |
| Symbol state | Fixed HUD crop, exact 3-by-3 array | Object encoder or slot representation |
| Temporal identity | Signature and position heuristics | Siamese tracker, recurrent slots, or object-centric encoder |
| Effect learning | Rule-based classifier and dictionaries | MLP/GNN transition and effect model |
| Memory | Python dictionaries and counters | Recurrent memory plus symbolic episodic store |
| Experiment selection | Handcrafted priority rules | Learned utility or policy/value network |
| Navigation | BFS | Retain BFS/A*, optionally neural heuristic |
| World model | Handcrafted reduced state | Learned object-centric ROM |
### CNN
A convolutional neural network should replace color-specific player and object detectors. Inputs may include:

- Current frame encoded as categorical one-hot channels.
- Previous frame.
- Temporal difference.
- Visit mask.
- Optional recent-action channels.

Outputs may include:

- Controllable-entity heatmap.
- Traversability mask.
- Objectness or interaction-affordance heatmap.
- HUD mask.
- Per-pixel or per-object embeddings.

A shallow CNN is appropriate for initial experiments because ARC visuals are small and geometrically structured.
### U-Net
A compact U-Net is suitable when spatially aligned segmentation masks are needed. It could jointly predict player, floor, obstacle, special-object, and interface regions. Skip connections preserve precise localization while deeper layers capture context.

The U-Net should not directly choose actions in the first neural version. It should produce a semantic scene representation for the existing causal learner and planner.
### Autoencoder or variational encoder
An autoencoder can learn a compressed latent representation, but reconstruction alone may preserve irrelevant pixels and ignore causal variables. If used, it should include auxiliary objectives for:

- Action-conditioned next-state prediction.
- Object position prediction.
- Progress-event prediction.
- Resource-change prediction.
- Contrastive separation of different mechanism effects.

A variational autoencoder is optional, not required. Discrete or object-factorized latents may suit the benchmark better than a single continuous vector.
### Recurrent model
A GRU or LSTM can integrate partially observed temporal information:

\[
h_{t+1}=\operatorname{GRU}(h_t,[E(I_t),a_t]).
\]

It may represent hidden phase, moving-object state, consumed mechanisms, or uncertain rules. A Transformer over a short trajectory is another option, but a GRU is a simpler first baseline for the small action space.

The recurrent state should supplement, not silently replace, explicit causal memory. Interpretability matters because debugging requires knowing which rule was inferred.
### Object-centric model
A more general system should represent entities as slots:

```text
entity ID
mask or bounding box
position
appearance embedding
velocity
controllability probability
mobility probability
interaction affordance
latent internal state
```

Candidate architectures include slot attention, recurrent object slots, or a supervised segmentation-and-tracking system. The key requirement is temporal identity: the same moving mechanism must remain one entity rather than becoming multiple static tiles.
### GNN
A graph neural network can model relations among entities. Nodes represent objects; edges represent spatial relations, contact, alignment, containment, or possible interaction. An action-conditioned message-passing model can estimate

\[
P(e_{t+1}^{1:N}\mid e_t^{1:N},a_t).
\]

This is particularly relevant when effects depend on relationships, such as touching a tile while carrying a symbol, crossing a gate in a certain phase, or colliding with a moving mechanism.
### Effect-model MLP
Before implementing a full GNN, a small MLP can classify local interaction effects. Possible inputs are:

- Embedding of the contacted object.
- Key or carried-object state before interaction.
- Action.
- Local spatial context.
- Recurrent memory state.

Outputs may include:

- Distribution over effect families.
- Predicted state delta.
- Confidence or epistemic uncertainty.
- Probability of progress, hazard, or resource gain.

Self-supervised labels can be generated from stable before-and-after differences, while ambiguous examples should retain uncertainty rather than receive forced labels.
### Neural world model
The long-term ROM can be a learned world model with object or latent state:

\[
z_{t+1}=F_\phi(z_t,a_t),
\]

\[
\hat o_{t+1}=D_\psi(z_{t+1}).
\]

Planning may use latent rollouts, model-predictive control, or search over symbolic abstractions extracted from the latent model. Ensemble disagreement can estimate uncertainty and prioritize informative experiments.
### Policy and value networks
A policy network should be introduced late, primarily for selecting experiments or proposing subgoals:

\[
\pi_\theta(o_i\mid z_t,\mathcal H_t).
\]

A value network may estimate success probability, remaining action cost, or risk. It should not replace exact BFS/A* navigation where a reliable graph is available.
### PPO or actor-critic
PPO or actor-critic is useful as a comparison baseline, not the recommended first architecture. Sparse rewards, changing game rules, and the need for test-time causal discovery make a purely model-free policy likely to overfit public games. A stronger experiment is to compare model-free RL against the neuro-symbolic architecture under held-out games.
## Why L5 remains unsolved
### Evidence from logs
The level-5 runs show that the agent successfully identifies `reshape`, `recolor`, `rotate_cw`, and resource mechanisms. It also detects numerous nonlocal player transitions. Nevertheless, it exhausts resources, repeats semantic states, loses lives, and reuses long plans that do not match the true dynamic mechanism.
### Primary hypothesis: mobile mechanisms become phantom tiles
The current `merge_scene` marks a visual signature as moving when its observed positions change and removes stale positions for signatures present in the current observation. This is a useful start, but the planner still fundamentally represents mechanisms by current cell and learned signature. It does not maintain a robust entity ID, trajectory, phase, or period.[^1]

A moving rotator can therefore be observed at several cells over time. If tracking fails because of occlusion, overlap, or signature changes, the map may contain stale or inconsistent interaction opportunities.
### Dynamic configuration is too brittle
Dynamic edges are indexed by an exact tuple of detected boxes. Small perceptual differences create distinct configurations, while semantically identical phases may not match. Conversely, incomplete configurations may cause unsafe reuse.
### Single-observation transitions
`transition_position` uses a stored dynamic edge as soon as it exists. Although `DynamicTransition` has an `observations` field, the planner does not require repeated confirmation or a confidence threshold.[^1]
### Gate identity contamination
Gate signatures hash a local patch around the line. Nearby moving objects or the player can alter this patch, changing identity even if the gate itself is unchanged.
### Hierarchical role assumption
The current order `reshape`, then `recolor`, then `rotate` is efficient when operators are independent. It can fail if an operator changes more than one factor, if contact with a moving mechanism is unavoidable, or if spatial routing forces temporary deviation.
### Resource estimate is myopic
The active-experiment logic reserves approximately the selected path plus a small margin. It does not calculate the cost of completing all remaining semantic operations and reaching a door under temporal constraints.
### Semantic cycle handling
Repeated states are counted but not used strongly enough to ban a failed plan schema, increase uncertainty, or trigger a different experiment.
## Recommended L5 redesign
### Separate static tiles and tracked entities
Replace the unified tile map with:

```python
StaticMechanism
TrackedEntity
ResourcePickup
GoalObject
```

A `TrackedEntity` should include:

```text
entity_id
appearance signature or embedding
current position
previous position
velocity
effect model
phase
estimated period
visibility confidence
last seen time
```
### Learn periodic dynamics
For each moving entity, fit candidate motion models:

- Stationary.
- Constant translation.
- Bounded oscillation.
- Cyclic trajectory.
- Action-conditioned movement.
- Contact-conditioned movement.

Maintain posterior scores or prediction errors. Do not insert a moving entity’s old positions into the permanent static map.
### Space-time planning
Extend the planning state to

\[
s_t=(p_t,k_t,r_t,m_t,q_t,i_t),
\]

where `q_t` is the phase/configuration of tracked dynamic entities and `i_t` is progress through an abstract operator sequence.

Every simulated action should advance both player and world phase. Waiting can be represented by a legal no-op if available or by a safe action cycle whose net spatial effect is zero.
### Confidence-gated edges
A nonlocal transition should become planning-safe only after repeated compatible observations. Suggested fields are:

```text
outcome counts
context cluster
observation count
confidence
last prediction error
```

Unknown or low-confidence transitions remain available for deliberate experiments but not for long execution plans.
### Abstract operator planning
First search in key space for a sequence of operators that transforms the current key into a target. Then search in space-time for a route that realizes that sequence. This decomposition avoids unrestricted joint search while retaining the ability to account for movement cost.
### Strict action verification
For each action, compare:

```text
predicted player position vs observed position
predicted key vs observed key
predicted resource vs observed resource
predicted dynamic phase vs observed phase
```

Any important mismatch should invalidate the remaining plan, update uncertainty, and trigger replanning.
### Key lock
Once the key exactly matches a door target, enter a protected execution mode. Transforming mechanisms become forbidden unless the model predicts that contact is unavoidable and reversible.
### Risk-sensitive resource planning
Plan with a safety constraint:

\[
C_{\text{remaining plan}}+C_{\text{uncertainty reserve}}\le r_t.
\]

If this inequality fails, prioritize a known refill before further experimentation.
## Generalization and “cheating”
Using only visible observations and legal actions is consistent with the intended agent interface. The current code does not read hidden environment state or the environment source during play. The official framework is specifically built for agents that consume frames and return actions.[^1][^7][^4]

However, the current agent is highly specialized. Hardcoded player colors, grid scale, HUD coordinates, symbol size, and effect ontology encode substantial prior knowledge about `ls20`. This is not equivalent to reading the answer, but it is not evidence of broad generalization.

A useful distinction is:

| Design choice | Interpretation |
|---|---|
| Hardcoded action sequence for a known level | Direct overfitting; unacceptable as general reasoning |
| Reading environment internals | Invalid observational evaluation |
| Handcrafted `ls20` pixel coordinates | Legitimate research baseline, but domain-specific |
| General priors such as object persistence and contact effects | Desirable inductive bias |
| Learned object and transition models from visible interaction | Target approach |
| Same unmodified agent tested on held-out games | Evidence of generalization |

Adding a neural network does not automatically remove specialization. A CNN trained only on `ls20` can memorize its colors and layouts. Generalization requires training and evaluation protocols that separate development games from held-out games.
## Hypothesis library for future games
The long-term hypothesis system should use parameterized program families rather than an ever-growing list of game-specific labels.
### Visual transformations
- Translation.
- Rotation by parameterized angle.
- Reflection.
- Recoloring or color permutation.
- Shape substitution.
- Toggle.
- Increment or cyclic state change.
- Copy.
- Merge.
- Split.
- Creation.
- Deletion.
### Spatial dynamics
- Collision.
- Conditional barrier.
- Teleportation.
- Transport by moving object.
- Push or pull.
- Sliding until collision.
- Periodic motion.
- Action-conditioned motion.
- Contact-conditioned motion.
### Resources
- Fixed action cost.
- State-dependent action cost.
- Refill.
- One-shot consumable.
- Life loss.
- Timer.
- Shared budget.
### Goal structures
- Match a visual target.
- Reach a location.
- Activate a set.
- Visit in sequence.
- Maintain a constraint.
- Avoid an object.
- Maximize or minimize a visible quantity.
- Trigger a positive terminal transition.

Each hypothesis should expose:

```text
preconditions
predicted effects
free parameters
confidence
evidence count
complexity cost
```

Model selection may minimize

\[
\mathcal L(h)=L_{\text{prediction}}(h)+\lambda C(h),
\]

favoring simple explanations that predict observations well.
## Recommended software architecture
Refactor the monolithic agent into modules:

```text
agent/
  controller.py
  observation_adapter.py
  perception/
    heuristic_ls20.py
    neural_segmenter.py
    entity_tracker.py
    hud_parser.py
  representation/
    semantic_state.py
    object_graph.py
    latent_rom.py
  learning/
    hypothesis_bank.py
    effect_learner.py
    transition_learner.py
    uncertainty.py
  planning/
    graph_search.py
    operator_planner.py
    temporal_planner.py
    risk_model.py
  memory/
    episodic.py
    semantic.py
  logging/
    trajectory.py
    diagnostics.py
  tests/
```

The `heuristic_ls20.py` module should preserve the successful domain-specific parser as a teacher and debugging oracle. New neural components can be swapped in behind common interfaces without discarding the working planner.
## Proposed interfaces
### Perception
```python
class PerceptionModel:
    def parse(self, frames, actions) -> BeliefState:
        ...
```
### Entity tracker
```python
class EntityTracker:
    def update(self, detections, previous_tracks) -> list[EntityTrack]:
        ...
```
### Transition learner
```python
class TransitionLearner:
    def update(self, state_before, action, state_after) -> None:
        ...

    def predict(self, state, action) -> TransitionDistribution:
        ...
```
### Experiment selector
```python
class ExperimentSelector:
    def choose(self, belief, hypotheses, goals) -> Subgoal:
        ...
```
### Planner
```python
class Planner:
    def plan(self, belief, models, goal) -> list[Action]:
        ...
```
### Verifier
```python
class Verifier:
    def compare(self, prediction, observation) -> ModelError:
        ...
```
## Data collection
The symbolic agent is valuable even before neural models are added because it can produce labeled interaction trajectories. Record every step as JSONL or HDF5:

```text
game and environment version
episode, level, timestep
frame before and after
action
parsed state before and after
object detections and tracks
contacted object
predicted effect distribution
observed effect
resource change
progress event
planner reason
model confidence
```

The official agents repository includes recording utilities and typed game structures, making trajectory capture compatible with the broader framework.[^8]

Do not store only successful episodes. Failed plans and model mismatches are essential training data for uncertainty estimation and recovery behavior.
## Training plan
### Stage A: symbolic teacher
- Stabilize the first four levels.
- Improve tracked dynamics enough to solve or clearly diagnose level 5.
- Export clean semantic labels.
- Create unit tests for rotations, recolors, reshapes, refills, blocked motion, nonlocal motion, and life reset.
### Stage B: neural perception
- Train a CNN or U-Net on heuristic pseudo-labels.
- Hold out layouts and colors.
- Measure player localization, traversability IoU, object detection, and HUD localization.
- Replace one heuristic detector at a time.
### Stage C: effect prediction
- Train an MLP or small relational model on before-action-after tuples.
- Predict effect family and semantic delta.
- Calibrate uncertainty.
- Compare against rule-based labels and exact transition dictionaries.
### Stage D: temporal tracking
- Train or implement entity association.
- Evaluate ID switches, position error, phase prediction, and occlusion recovery.
- Integrate tracked entities into the space-time planner.
### Stage E: learned ROM
- Train action-conditioned latent dynamics.
- Require the latent to predict objects, resources, progress, and uncertainty.
- Compare global-vector, recurrent, and object-centric representations.
### Stage F: learned experiment policy
- Train a policy or value model to rank candidate experiments.
- Preserve hard safety and legality constraints.
- Compare learned ranking against distance-only and information-gain heuristics.
### Stage G: held-out games
- Freeze code and hyperparameters.
- Evaluate on public games excluded from development.
- Record how many assumptions require no changes.
- Treat any game-specific patch after evaluation as a new development iteration, not a held-out result.
## Experimental matrix
| Agent | Perception | Effects | Planning | Purpose |
|---|---|---|---|---|
| Random | None | None | None | Minimum baseline |
| Frontier | Heuristic map | None | Graph exploration | Spatial baseline |
| Salient nearest | Heuristic salience | None | BFS | Tests directed interaction |
| Current agent | `ls20` heuristics | Symbolic causal models | Joint BFS | Present baseline |
| Neural perception | CNN/U-Net | Symbolic models | Joint BFS | Tests visual generalization |
| Neural effects | CNN/U-Net | MLP/GNN | Joint BFS | Tests learned causal prediction |
| Learned ROM | Object/latent encoder | Latent dynamics | MPC/search | Tests model-based learning |
| Model-free RL | CNN | Implicit | PPO/actor-critic | Comparison baseline |
## Metrics
Track at least:

- Levels completed.
- Whole-game completion rate.
- Actions per completed level.
- Relative action efficiency where available.
- Exploration actions before first valid model.
- Number of unnecessary interactions.
- Effect-classification accuracy.
- Semantic-state prediction accuracy.
- Dynamic-entity tracking accuracy.
- Calibration of prediction confidence.
- Number of replans after model mismatch.
- Recovery rate after incorrect hypotheses.
- Resource wasted during failed experiments.
- Performance on held-out levels and games.
- Variance across seeds.
## Ablations
Recommended ablations include:

- Remove salience and use random exploration.
- Remove persistent cross-level memory.
- Remove exact key transition mappings.
- Remove rotational augmentation of shape models.
- Remove dynamic configuration from planning.
- Remove resource-aware planning.
- Remove closed-loop replanning.
- Replace object-centric state with a global latent vector.
- Replace symbolic planning with a policy network.
- Replace learned uncertainty with maximum-probability predictions.
## Testing requirements
### Unit tests
- Each rotation classifier.
- Recolor with unchanged occupancy.
- Reshape with changed occupancy.
- Resource refill detection after normal step-cost correction.
- Exact key-goal equality.
- Door blocked when key is wrong.
- Consumed pickup bit mask.
- Dynamic transition conditioned on configuration.
- Life reset and memory separation.
- Moving signature stale-position removal.
### Property tests
- Four clockwise rotations restore the key.
- Canonical shape is rotation-invariant.
- Resource never becomes negative in simulated states.
- Reconstructed BFS paths contain only legal transitions.
- A key-lock plan never applies a known transformation.
### Replay tests
Save deterministic trajectories for levels already solved and rerun perception and learning offline. This prevents a change intended for L5 from silently regressing L1–L4.
## Immediate engineering priorities
1. Preserve the current `agent.py` as a tagged baseline.
2. Add trajectory recording before changing behavior.
3. Build regression replays for L1–L4.
4. Refactor perception from planning without changing outputs.
5. Replace mobile tile handling with persistent entity tracks.
6. Normalize gate identity independently of surrounding pixels.
7. Add confidence thresholds for dynamic edges.
8. Implement periodic phase estimation.
9. Add abstract key-operator planning followed by space-time realization.
10. Add strict prediction verification and plan invalidation.
11. Re-evaluate L5 over multiple seeds or deterministic runs.
12. Begin CNN/U-Net perception only after labels and interfaces are stable.
## Guidance for another AI
An AI continuing the work should follow these constraints:

- Do not claim neural networks are already implemented.
- Do not invent successful runs.
- Do not hardcode an action sequence, level number, target vector, or known solution.
- Do not inspect environment source or hidden state for decision-making.
- Preserve all behavior that already solves L1–L4 unless a regression test justifies a change.
- Request the exact latest `agent.py`, logs, `memory.json`, and environment version before producing a drop-in replacement.
- Distinguish a safe patch from a complete replacement.
- Make one architectural change at a time when possible.
- Include diagnostic logs that expose predicted state, observed state, confidence, and replan cause.
- Design abstractions that can later accept neural outputs.
## Recommended AI work request
The following prompt can be sent with this document and the latest source files:

```text
You are continuing development of an observation-only ARC-AGI-3 agent.
Read the attached project handoff and inspect the exact current agent.py.
Do not assume neural modules are implemented. Do not read or exploit game
source code, hidden state, hardcoded level numbers, target arrays, or known
action sequences.

First produce:
1. A precise call graph and state-flow description of the current code.
2. A diagnosis supported by specific functions and data structures.
3. A minimal patch plan that preserves L1-L4 behavior.
4. Unit and replay tests that fail before the patch and pass afterward.

Primary target: represent mobile/periodic mechanisms as tracked entities,
not persistent static tiles; use confidence-gated, phase-conditioned
transitions in planning; validate each predicted action against the next
observation.

Only after the design review, produce code. Clearly separate implemented
changes, assumptions, and unverified hypotheses. Include commands to run
unit tests and a complete ls20 evaluation.
```
## Questions for future AI discussions
- Is the L5 mechanism truly periodic, action-conditioned, or contact-conditioned?
- Can its phase be inferred from visible frames alone?
- Does the interaction effect occur before or after mechanism movement?
- Are multiple apparent rotators one tracked entity or several synchronized entities?
- Which visual features remain stable through overlap and occlusion?
- Can gate transitions be represented relative to local geometry rather than absolute position?
- Does the sequential shape-color-rotation decomposition remain valid on L6 and L7?
- What is the minimum sufficient latent state for prediction?
- Can the agent infer controllability without known player colors?
- Which learned modules transfer to a second public game without retraining?
## Final project framing
The current system should be described as an **interpretable symbolic causal baseline with a handcrafted reduced-order state**, not as a completed neural agent. Its main contribution is the architecture and experimental methodology:

- Explore semantically meaningful objects instead of acting randomly.
- Learn causal effects from interventions.
- Compress visual observations into a planning state.
- Separate world-model learning from action search.
- Preserve knowledge across related levels.
- Replan after every observation.
- Progressively replace domain-specific perception and transition rules with neural, object-centric, uncertainty-aware models.

That progression provides a coherent course project: begin with a transparent system that demonstrates active reasoning on `ls20`, quantify its limitations, add neural perception and dynamics in controlled stages, and evaluate generalization by freezing the agent and testing it on games excluded from development.

---

## References

1. [agent.py](arc-special-tile-agent/agent.py) (original upload link removed: it carried a temporary AWS token)

2. [arcprize/ARC-AGI-3-Agents - GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) - 1. Get an API key from the ARC-AGI-3 Website and set it as an environment variable in your .env file...

3. [ARC-AGI-3-Kaggle-Starter/README.md at main - GitHub](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter/blob/main/README.md) - This is a starter kit for the ARC Prize 2026 — ARC-AGI-3 competition.

4. [ARC-AGI-3-Kaggle-Starter/agent/my_agent.py at main · arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter/blob/main/agent/my_agent.py) - Contract (enforced by the ARC-AGI-3-Agents framework):

5. [ARC-AGI-3](https://arcprize.org/arc-agi/3) - ARC-AGI-3 is an interactive reasoning benchmark which challenges AI agents to explore novel environm...

6. README-2.md (the earlier salient-tile prototype README; original upload link removed: it carried a temporary AWS token)

7. [docs/agents-quickstart.mdx at 04ce3d1dd4deebe10f01cbffce0483c4e95c4302 · arcprize/docs](https://github.com/arcprize/docs/blob/04ce3d1d/agents-quickstart.mdx?plain=1) - The following is based off the [ARC-AGI-3 Agents repo](https://github.com/arcprize/ARC-AGI-3-Agents)...

8. [llms.txt - arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents/blob/main/llms.txt) - The core agent framework lives in [`agents/`](https://github.com/arcprize/ARC-AGI-3-Agents/tree/main...

