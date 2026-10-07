"""Conservative well-mixed particle transport, not CFD or cleanroom certification."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Room:
    name: str
    volume_m3: float
    pressure_pa: float
    supply_m3_h: float
    exhaust_m3_h: float
    generation_particles_h: tuple = (0., 0.)
    deposition_per_h: tuple = (.1, .5)


def simulate(rooms, connections, initial_counts, duration_h=.5, step_h=1 / 3600):
    if not rooms or len({room.name for room in rooms}) != len(rooms):
        raise ValueError("Require uniquely named rooms.")
    if not all(math.isfinite(value) and value > 0 for value in (duration_h, step_h)):
        raise ValueError("Duration and time step must be finite and positive.")
    if math.ceil(duration_h / step_h) > 200000:
        raise ValueError("Simulation exceeds the bounded teaching workload.")
    for room in rooms:
        scalars = (room.volume_m3, room.pressure_pa, room.supply_m3_h, room.exhaust_m3_h, *room.generation_particles_h, *room.deposition_per_h)
        if not all(math.isfinite(value) for value in scalars) or room.volume_m3 <= 0 or min(room.supply_m3_h, room.exhaust_m3_h, *room.generation_particles_h, *room.deposition_per_h) < 0:
            raise ValueError("Room parameters must be finite with positive volume and nonnegative rates.")
        if len(room.generation_particles_h) != 2 or len(room.deposition_per_h) != 2:
            raise ValueError("Use two disjoint size bins: [0.5,5) µm and [5,infinity) µm.")
    index = {room.name: i for i, room in enumerate(rooms)}
    flows = []
    for left, right, conductance in connections:
        if left not in index or right not in index or left == right or not math.isfinite(conductance) or conductance < 0:
            raise ValueError("Invalid room connection or conductance.")
        a, b = index[left], index[right]
        difference = rooms[a].pressure_pa - rooms[b].pressure_pa
        source, target = (a, b) if difference >= 0 else (b, a)
        flows.append((source, target, conductance * abs(difference)))
    air_balance = np.array([room.supply_m3_h - room.exhaust_m3_h for room in rooms])
    for source, target, flow in flows:
        air_balance[source] -= flow
        air_balance[target] += flow
    if not np.allclose(air_balance, 0, atol=1e-9, rtol=0):
        raise ValueError("Prescribed flows violate constant-volume air balance; revise supply/exhaust or pressure assumptions.")
    counts = np.asarray(initial_counts, dtype=float).copy()
    if counts.shape != (len(rooms), 2) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Initial state must be finite nonnegative particle counts, one row per room and two disjoint bins.")
    initial_total = counts.sum(axis=0)
    generated, removed, transferred = np.zeros(2), np.zeros(2), np.zeros((len(flows), 2))
    volumes = np.array([room.volume_m3 for room in rooms])
    exhaust = np.array([room.exhaust_m3_h for room in rooms])
    deposition = np.array([room.deposition_per_h for room in rooms])
    generation = np.array([room.generation_particles_h for room in rooms])
    outgoing = exhaust.copy()
    for source, _, flow in flows:
        outgoing[source] += flow
    depletion_rate = outgoing[:, None] / volumes[:, None] + deposition
    if (min(step_h, duration_h) * depletion_rate > 1).any():
        raise ValueError("Time step violates the positivity bound; reduce step_h rather than clipping negative counts.")
    frames = [{"time_h": 0., "counts": counts.tolist()}]
    stride = max(1, math.ceil(duration_h / step_h) // 60)
    for step in range(math.ceil(duration_h / step_h)):
        dt = min(step_h, duration_h - step * step_h)
        losses = (exhaust[:, None] / volumes[:, None] + deposition) * counts * dt
        additions = generation * dt
        delta = additions - losses
        for edge, (source, target, flow) in enumerate(flows):
            movement = flow * dt * counts[source] / volumes[source]
            delta[source] -= movement
            delta[target] += movement
            transferred[edge] += movement
        counts += delta
        generated += additions.sum(axis=0)
        removed += losses.sum(axis=0)
        if not np.isfinite(counts).all() or (counts < -1e-9).any():
            raise FloatingPointError("Particle state violated finiteness or positivity.")
        if (step + 1) % stride == 0 or step == math.ceil(duration_h / step_h) - 1:
            frames.append({"time_h": min((step + 1) * step_h, duration_h), "counts": counts.tolist()})
    residual = initial_total + generated - removed - counts.sum(axis=0)
    concentration = counts / volumes[:, None]
    return {"model": "well-mixed, constant-volume, clean-supply, fixed-pressure compartment model",
            "size_bins_um": ["0.5 <= d < 5", "d >= 5"], "duration_h": duration_h, "step_h": step_h,
            "rooms": [{"name": room.name, "volume_m3": room.volume_m3, "pressure_pa": room.pressure_pa,
                       "supply_m3_h": room.supply_m3_h, "exhaust_m3_h": room.exhaust_m3_h,
                       "supply_air_changes_per_h": room.supply_m3_h / room.volume_m3,
                       "final_particles_m3_ge_0_5": float(concentration[i].sum()), "final_particles_m3_ge_5": float(concentration[i, 1])}
                      for i, room in enumerate(rooms)],
            "flows": [{"source": rooms[source].name, "target": rooms[target].name, "air_m3_h": flow,
                       "transported_particle_counts_by_bin": transferred[i].tolist()} for i, (source, target, flow) in enumerate(flows)],
            "initial_total_by_bin": initial_total.tolist(), "generated_total_by_bin": generated.tolist(),
            "removed_total_by_bin": removed.tolist(), "final_total_by_bin": counts.sum(axis=0).tolist(),
            "conservation_residual_by_bin": residual.tolist(), "frames": frames,
            "limits": "No velocity field, laminar-flow proof, viable-organism model, door transient, filter-leak model, or ISO classification decision."}
