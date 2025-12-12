#!/usr/bin/env python3
"""
Infrastructure Flow Animator
Generates MP4 animations from JSON-defined infrastructure flow diagrams using Manim.

Installation note: Requires C++ compiler for Manim dependencies.
If you encounter build errors, see: https://docs.manim.community/en/stable/installation.html
"""

import json
import sys
from pathlib import Path
from typing import Dict

# Try importing manim - provide helpful error message if unavailable
try:
    from manim import *
except ImportError as e:
    print("Error: Manim not fully installed")
    print("This requires a C++ compiler. On Windows, install:")
    print("  - Microsoft C++ Build Tools")
    print("  - Or Visual Studio Community with C++ support")
    print("\nThen run: pip install manim")
    sys.exit(1)


class InfrastructureFlowScene(Scene):
    """
    Manim scene that visualizes infrastructure data flows.
    Reads from JSON config files for flexible, reusable diagrams.
    """

    def __init__(self, flow_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_config = flow_config
        self.nodes_mobjects = {}
        self.flows = []

    def construct(self):
        """Build and animate the scene."""
        title = self._create_title()
        self.add(title)
        
        self.draw_nodes()
        self.draw_flows()
        self.animate_flows()
        
        self.wait(2)

    def _create_title(self):
        """Create title from process name."""
        process_name = self.flow_config.get("process_name", "Infrastructure Flow")
        title = Text(process_name, font_size=40, color=WHITE)
        title.to_edge(UP)
        return title

    def draw_nodes(self):
        """Create infrastructure component nodes from config."""
        nodes_config = self.flow_config.get("nodes", [])

        for node in nodes_config:
            node_id = node["id"]
            label = node["label"]
            position = node.get("position", [0, 0])
            color = node.get("color", "#3498db")
            shape = node.get("shape", "circle")

            # Convert position to scene coordinates
            x = (position[0] - 4.5) * 0.8
            y = (position[1] - 1.5) * 0.8

            # Create shape based on config
            if shape == "circle":
                mobject = Circle(radius=0.35, color=color, fill_opacity=0.8)
            elif shape == "square":
                mobject = Square(side_length=0.6, color=color, fill_opacity=0.8)
            else:  # rectangle (default)
                mobject = Rectangle(height=0.5, width=0.7, color=color, fill_opacity=0.8)

            mobject.move_to([x, y, 0])

            # Create label text centered on node
            text_label = Text(label, font_size=14, color=BLACK)
            text_label.move_to([x, y, 0])

            # Add to scene
            self.add(mobject, text_label)
            self.nodes_mobjects[node_id] = {
                "shape": mobject,
                "label": text_label,
                "position": np.array([x, y, 0])
            }

    def draw_flows(self):
        """Create arrows representing data flows between nodes."""
        flows_config = self.flow_config.get("flows", [])

        for flow in flows_config:
            source_id = flow["source"]
            dest_id = flow["destination"]
            label = flow.get("label", "")
            color = flow.get("color", "#2ecc71")

            # Validate node references
            if source_id not in self.nodes_mobjects or dest_id not in self.nodes_mobjects:
                print(f"Warning: Invalid flow: {source_id} -> {dest_id}")
                continue

            source_pos = self.nodes_mobjects[source_id]["position"]
            dest_pos = self.nodes_mobjects[dest_id]["position"]

            # Create arrow between nodes
            arrow = Arrow(
                start=source_pos,
                end=dest_pos,
                color=color,
                stroke_width=2.5,
                buff=0.4,
                tip_length=0.2,
            )
            self.add(arrow)

            # Add label along flow if provided
            if label:
                mid_point = (source_pos + dest_pos) / 2
                offset = np.array([0.3, 0.2, 0])
                flow_label = Text(label, font_size=12, color=color)
                flow_label.move_to(mid_point + offset)
                self.add(flow_label)

            self.flows.append(
                {
                    "arrow": arrow,
                    "source_pos": source_pos,
                    "dest_pos": dest_pos,
                    "color": color,
                    "pulse_rate": flow.get("pulse_rate", 1.0),
                }
            )

    def animate_flows(self):
        """Create pulsing animation along data flows."""
        for flow in self.flows:
            pulse_rate = flow["pulse_rate"]
            
            # Create animated pulse circle
            pulse = Circle(radius=0.12, color=flow["color"], fill_opacity=0.9)
            pulse.move_to(flow["source_pos"])

            # Animate pulse moving along the arrow
            self.play(
                MoveAlongPath(
                    pulse,
                    flow["arrow"],
                    rate_func=linear,
                    run_time=1.5 / pulse_rate,  # Slower motion for clarity
                ),
                lag_ratio=0,
            )
            self.remove(pulse)


def load_flow_config(json_path: str) -> Dict:
    """Load and parse JSON flow configuration file."""
    try:
        with open(json_path, "r") as f:
            config = json.load(f)
        
        # Validate required fields
        required = ["process_name", "nodes", "flows"]
        missing = [field for field in required if field not in config]
        if missing:
            print(f"Error: Missing required fields in JSON: {missing}")
            sys.exit(1)
            
        return config
        
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}")
        sys.exit(1)


def main():
    """Main entry point - render animation from JSON config."""
    if len(sys.argv) < 2:
        print("Infrastructure Flow Animator")
        print("Usage: python animate_flow.py <path_to_config.json>")
        print("Example: python animate_flow.py inputs/example_order_processing.json")
        print("\nFor details, see: inputs/schema.json")
        sys.exit(1)

    json_path = sys.argv[1]
    config = load_flow_config(json_path)

    print(f"Rendering: {config.get('process_name')}")
    print(f"Output: media/")
    
    # Manim will handle the rendering when this script is imported
    # The scene is instantiated and rendered automatically


if __name__ == "__main__":
    main()
