# Tom.exe vs Jerry.exe — The Great AI Chase

A playful robotics project built with NVIDIA Isaac Sim 6.2 and Isaac Lab 3.0.
Original geometric characters chase through a tiny obstacle-filled room while
we compare direct pursuit, A* path planning, and a PPO policy honestly.

## Current milestone

- [x] NVIDIA Brev L40S provisioned and secured
- [x] Isaac Sim 6.2 compatibility check passed
- [x] Full headless Isaac Sim runtime reached `app ready`
- [x] Stage 2 primitive room, Tom, and Jerry
- [x] Authenticated Brev viewer endpoint configured
- [ ] Browser stream visually verified
- [ ] Stage 3 direct-pursuit baseline
- [ ] Stage 4 A* planner
- [ ] Stage 5 Isaac Lab PPO environment and training attempt
- [ ] Stage 6 reproducible comparison
- [ ] Stage 7 portfolio documentation and demo

## Design principles

- Use only original geometric characters and simple materials.
- Keep path planning distinct from reinforcement learning.
- Record fixed seeds and real metrics; never invent convergence.
- Keep cloud endpoints private and avoid idle GPU cost.
- Build the simplest working version before adding complexity.

## Repository layout

```text
src/tom_jerry_chase/   Isaac Sim and Isaac Lab source code
infra/                 Docker Compose overrides for cloud streaming
docs/                  Setup and architecture notes
experiments/           Reproducible configurations and CSV results
outputs/               User-facing deliverables
```

## Verified Stage 2 result

`src/tom_jerry_chase/scene_stage2.py` builds a 12 m × 9 m room from OpenUSD
primitives. It includes four collidable walls, two route-splitting obstacles,
a blue capsule cat, an orange sphere mouse, rigid bodies, gravity, lighting,
and a camera.

The scene was executed inside the official Isaac Sim 6.2 container on the Brev
L40S. Its self-check reported:

```text
STAGE2_OK path=/workspace/output/tom_jerry_stage2.usda prims=37 physics_hz=60
```

The generated USDA was then checked for `/World/PhysicsScene`, `/World/Tom`,
and `/World/Jerry`. Detailed setup and repeatable run commands will continue to
be added as each milestone is verified on the Brev GPU rather than written
speculatively.

## Secure Brev streaming

The official Isaac Sim Docker Compose viewer uses three ports:

| Port | Protocol | Purpose |
| --- | --- | --- |
| 8210 | TCP | Web viewer |
| 49100 | TCP | WebRTC signaling |
| 47998 | UDP | WebRTC media |

The web page is exposed through a Brev-authenticated HTTPS endpoint. The raw
WebRTC ports are restricted on the VM itself to the current client IP. Do not
leave unauthenticated streaming ports open without that host firewall.

Create the authenticated viewer route:

```bash
brev ports open tom-exe-jerry-exe 8210 --protocol http --json
```

Set `WEB_VIEWER_ALLOWED_HOST` to the hostname returned by Brev, without the
`https://` prefix. The override passes that exact hostname to Vite instead of
allowing every host:

```bash
WEB_VIEWER_ALLOWED_HOST="<authenticated-viewer-host>" \
ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0 \
ISAACSIM_HOST="<brev-public-ip>" \
ISAAC_SIM_DATA=/home/ubuntu/docker/isaac-sim \
docker compose -p tomjerry \
  -f /home/ubuntu/isaacsim-streaming/tools/docker/docker-compose.yml \
  -f /home/ubuntu/tom-exe-vs-jerry-exe/infra/docker-compose.stream.override.yml \
  up --build -d
```

The temporary `iptables` rules used during development allow only
`<current-client-ip>/32` to the three streaming ports, followed by a drop rule
for every other source. These rules are deliberately not persisted across a
VM reboot; verify and reapply them whenever the instance or client network
changes.

The Brev L40S used for this project costs `$1.79/hour` in the selected region.
Stop the VM whenever development pauses so credits are not consumed while idle.
