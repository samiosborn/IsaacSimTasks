# scripts/single_franka.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Parse CLI args first so we can configure the sim cleanly
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="Differential inverse kinematics method",
)
parser.add_argument("--episodes", type=int, default=1, help="Number of pick-and-place episodes to run")
parser.add_argument("--keep-open", action="store_true", help="Keep the window open after finishing")
args, _ = parser.parse_known_args()

# Isaac Sim requires SimulationApp to be created before other isaac/omni imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager

# Ensure the project root is importable so we can do `from src...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import after SimulationApp is instantiated
from src.controller.pick_place import FrankaPickPlace


def main():
    print("Starting Simple Franka Pick-and-Place Demo")

    # Select the physics device backend
    SimulationManager.set_physics_sim_device(args.device)

    # First update lets the app finish initialising
    simulation_app.update()

    # Build the task and scene
    pick_place = FrankaPickPlace()
    pick_place.setup_scene()

    # Start the timeline so physics steps advance
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    episodes_done = 0
    reset_needed = True

    # Main loop: step simulation, run controller, reset between episodes
    while simulation_app.is_running():
        if SimulationManager.is_simulating():
            if reset_needed:
                pick_place.reset()
                reset_needed = False

            pick_place.forward(args.ik_method)

            if pick_place.is_done():
                episodes_done += 1
                print(f"Episode {episodes_done} done")

                if episodes_done >= args.episodes:
                    break

                reset_needed = True

        # This drives rendering and extension updates
        simulation_app.update()

    # Useful for inspecting the final state without immediately exiting
    if args.keep_open:
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        simulation_app.close()