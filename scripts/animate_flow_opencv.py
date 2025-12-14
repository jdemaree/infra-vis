#!/usr/bin/env python3
"""
Infrastructure Flow Animator (OpenCV + Plotly version)
Generates MP4 animations from JSON-defined infrastructure flow diagrams.

This version doesn't require Manim's heavy C++ dependencies.
Instead, it uses Plotly for visualization and OpenCV for video encoding.
"""

import json
import sys
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import textwrap

import plotly.graph_objects as go

# Try to import cv2, provide helpful message if not available
try:
    import cv2
except ImportError:
    print("Note: cv2 (OpenCV) not installed. Install with: pip install opencv-python")
    print("For now, we'll generate static images instead of videos.")
    cv2 = None


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


class InfrastructureFlowVisualizer:
    """Create infrastructure flow visualizations and animations."""

    def __init__(self, config: Dict):
        self.config = config
        self.process_name = config.get("process_name", "Infrastructure Flow")
        self.nodes = config.get("nodes", [])
        self.flows = config.get("flows", [])

    def create_static_visualization(self, output_path: str):
        """Create a static Plotly visualization as HTML/SVG."""
        # Build node positions and properties
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in self.nodes:
            position = node.get("position", [0, 0])
            node_x.append(position[0])
            node_y.append(position[1])
            node_text.append(node["label"])
            node_color.append(node.get("color", "#3498db"))
            node_size.append(20)  # Node size for scatter plot

        # Build edge traces for flows
        edge_traces = []
        for flow in self.flows:
            # Support either simple source/destination flows or multi-node `route` flows
            if "route" in flow:
                # build polyline through the route node ids
                xs = []
                ys = []
                for nid in flow.get("route", []):
                    node_obj = next((n for n in self.nodes if n["id"] == nid), None)
                    if node_obj:
                        pos = node_obj.get("position", [0, 0])
                        xs.append(pos[0])
                        ys.append(pos[1])

                if xs and ys:
                    edge_trace = go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(width=2, color=flow.get("color", "#2ecc71"), dash="dot"),
                        hoverinfo="text",
                        text=[flow.get("label", "flow")],
                        showlegend=False,
                    )
                    edge_traces.append(edge_trace)
            else:
                source_id = flow.get("source")
                dest_id = flow.get("destination")

                # Find source and destination nodes
                source_node = next(
                    (n for n in self.nodes if n["id"] == source_id), None
                )
                dest_node = next((n for n in self.nodes if n["id"] == dest_id), None)

                if source_node and dest_node:
                    source_pos = source_node.get("position", [0, 0])
                    dest_pos = dest_node.get("position", [0, 0])

                    edge_trace = go.Scatter(
                        x=[source_pos[0], dest_pos[0]],
                        y=[source_pos[1], dest_pos[1]],
                        mode="lines",
                        line=dict(width=2, color=flow.get("color", "#2ecc71"), dash="dot"),
                        hoverinfo="text",
                        text=[flow.get("label", "flow")],
                        showlegend=False,
                    )
                    edge_traces.append(edge_trace)

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
        )

        # Build figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=f"{self.process_name}<br><sub>Infrastructure Flow Visualization</sub>",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1200,
            height=800,
        )

        # Save as HTML and static image
        html_path = output_path.replace(".mp4", ".html")
        fig.write_html(html_path)
        print(f"Saved interactive visualization: {html_path}")

        # Try to save as PNG
        try:
            png_path = output_path.replace(".mp4", ".png")
            fig.write_image(png_path)
            print(f"Saved static image: {png_path}")
        except Exception as e:
            print(f"Note: Could not save PNG (requires kaleido): {e}")

    def _render_frame(self, frame_num: int, total_frames: int, temp_dir: Path) -> tuple:
        """Render a single frame and return (frame_num, path)."""
        # Create figure for this frame
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in self.nodes:
            position = node.get("position", [0, 0])
            node_x.append(position[0])
            node_y.append(position[1])
            node_text.append(node["label"])
            node_color.append(node.get("color", "#3498db"))
            node_size.append(20)

        # Create edge traces with animation
        edge_traces = []
        for flow in self.flows:
            # Support either simple source/destination flows or multi-node `route` flows
            if "route" in flow:
                # draw each consecutive segment in the route with the pulse effect
                route_nodes = [nid for nid in flow.get("route", [])]
                for i in range(len(route_nodes) - 1):
                    n1 = next((n for n in self.nodes if n["id"] == route_nodes[i]), None)
                    n2 = next((n for n in self.nodes if n["id"] == route_nodes[i + 1]), None)
                    if not n1 or not n2:
                        continue
                    source_pos = n1.get("position", [0, 0])
                    dest_pos = n2.get("position", [0, 0])

                    pulse_rate = flow.get("pulse_rate", 1)
                    pulse_phase = (frame_num * pulse_rate / total_frames) % 1
                    width = 2 + 1 * abs(np.sin(pulse_phase * np.pi))  # Reduced width variation for muted effect

                    edge_trace = go.Scatter(
                        x=[source_pos[0], dest_pos[0]],
                        y=[source_pos[1], dest_pos[1]],
                        mode="lines",
                        line=dict(width=width, color=flow.get("color", "#2ecc71"), dash="dot"),
                        showlegend=False,
                    )
                    edge_traces.append(edge_trace)
            else:
                source_id = flow.get("source")
                dest_id = flow.get("destination")

                source_node = next(
                    (n for n in self.nodes if n["id"] == source_id), None
                )
                dest_node = next((n for n in self.nodes if n["id"] == dest_id), None)

                if source_node and dest_node:
                    source_pos = source_node.get("position", [0, 0])
                    dest_pos = dest_node.get("position", [0, 0])

                    # Vary width based on animation frame to create pulse effect
                    pulse_rate = flow.get("pulse_rate", 1)
                    pulse_phase = (frame_num * pulse_rate / total_frames) % 1
                    width = 2 + 1 * abs(np.sin(pulse_phase * np.pi))  # Reduced width variation

                    edge_trace = go.Scatter(
                        x=[source_pos[0], dest_pos[0]],
                        y=[source_pos[1], dest_pos[1]],
                        mode="lines",
                        line=dict(width=width, color=flow.get("color", "#2ecc71"), dash="dot"),
                        showlegend=False,
                    )
                    edge_traces.append(edge_trace)

        # If a flow defines a `route` and `moving: true`, prepare route geometry
        route_flow = None
        for f in self.flows:
            if f.get("route") and f.get("moving"):
                route_flow = f
                break

        route_positions = []
        seg_lengths = []
        total_route_length = 0.0
        route_entries = []
        if route_flow:
            # Support route entries as either simple node-id strings or
            # objects {"id": "node_id", "description": "..."}
            raw_route = route_flow.get("route", [])
            for item in raw_route:
                if isinstance(item, dict):
                    nid = item.get("id")
                    route_entries.append(item)
                else:
                    nid = item
                    route_entries.append({"id": nid})

                node_obj = next((n for n in self.nodes if n["id"] == nid), None)
                if node_obj:
                    route_positions.append(tuple(node_obj.get("position", [0, 0])))

            # Compute per-segment lengths
            for i in range(len(route_positions) - 1):
                x1, y1 = route_positions[i]
                x2, y2 = route_positions[i + 1]
                seg_len = float(np.hypot(x2 - x1, y2 - y1))
                seg_lengths.append(seg_len)
                total_route_length += seg_len

        # Node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=node_size, color=node_color, line=dict(width=2)),
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])

        # Add single moving marker following the defined route (if any)
        if route_flow and route_positions and total_route_length > 0:
            speed = float(route_flow.get("speed", 1.0))
            progress = ((frame_num / total_frames) * speed) % 1.0
            distance_along = progress * total_route_length

            # locate segment
            acc = 0.0
            seg_idx = 0
            local_t = 0.0
            for i, seg_len in enumerate(seg_lengths):
                if acc + seg_len >= distance_along:
                    seg_idx = i
                    local_t = (distance_along - acc) / seg_len if seg_len > 0 else 0.0
                    break
                acc += seg_len

            x1, y1 = route_positions[seg_idx]
            x2, y2 = route_positions[seg_idx + 1]
            mx = x1 + (x2 - x1) * local_t
            my = y1 + (y2 - y1) * local_t

            moving_trace = go.Scatter(
                x=[mx],
                y=[my],
                mode="markers",
                marker=dict(size=20, color=route_flow.get("color", "#e74c3c"), line=dict(width=2, color="white")),
                hoverinfo="none",
                showlegend=False,
            )
            fig.add_traces([moving_trace])

            # Determine the current stage/node to display in the context box
            # based on the route progress (segment index + local t) so the
            # description exactly follows the dot along the route.
            stage_node_obj = None
            try:
                # seg_idx and local_t were computed above when locating the marker
                # Choose the node index: if the marker is in the first half of the
                # segment, show the start node; otherwise show the end node.
                node_index = seg_idx if local_t < 0.5 else seg_idx + 1
                # Clamp index into valid range
                node_index = max(0, min(node_index, len(route_entries) - 1))
                route_entry = route_entries[node_index]
                node_id_for_stage = route_entry.get("id")
                stage_node_obj = next((n for n in self.nodes if n["id"] == node_id_for_stage), None)
            except Exception:
                stage_node_obj = None

            # Prepare context text (prefer `description` on the stage node)
            if stage_node_obj:
                stage_title = stage_node_obj.get("label", stage_node_obj.get("id", "Stage"))
                # Prefer per-route-step description when provided
                stage_desc = route_entry.get("description") if route_entry.get("description") else stage_node_obj.get("description", stage_node_obj.get("note", ""))
                if not stage_desc:
                    stage_desc = route_flow.get("label", "")
            else:
                stage_title = self.process_name
                stage_desc = ""

            # Prepare wrapped plain-text description (no HTML) and limit length
            raw_desc = str(stage_desc or "")
            # Truncate very long descriptions to avoid overflowing the box
            max_chars = 800
            if len(raw_desc) > max_chars:
                raw_desc = raw_desc[: max_chars - 3] + "..."

            wrapped = textwrap.fill(raw_desc, width=40)
            desc_text = f"{stage_title}\n{wrapped}" if wrapped else stage_title

            # Add a static context box in the top-left (paper coordinates)
            box_x0 = 0.02
            box_x1 = 0.28
            box_y1 = 0.98
            box_y0 = 0.70

            fig.update_layout(shapes=[
                dict(
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=box_x0,
                    y0=box_y0,
                    x1=box_x1,
                    y1=box_y1,
                    line=dict(color="#444", width=1),
                    fillcolor="rgba(255,255,255,0.95)",
                    layer="above",
                )
            ])

            fig.add_annotation(
                text=desc_text,
                xref="paper",
                yref="paper",
                x=(box_x0 + 0.02),
                y=(box_y1 - 0.02),
                showarrow=False,
                align="left",
                xanchor="left",
                yanchor="top",
                font=dict(size=12, color="#111"),
                borderpad=4,
            )

        fig.update_layout(
            title=self.process_name,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1280,
            height=720,
        )

        # Save frame as PNG
        frame_path = temp_dir / f"frame_{frame_num:04d}.png"
        fig.write_image(str(frame_path))
        return (frame_num, str(frame_path))


    def _calculate_animation_duration(self) -> int:
        """Calculate animation duration (frames) based on route length."""
        # Find the moving route flow
        route_flow = None
        for f in self.flows:
            if f.get("route") and f.get("moving"):
                route_flow = f
                break

        if not route_flow:
            return 120  # Default 120 frames if no route

        # Calculate total route length
        route_nodes = route_flow.get("route", [])
        total_segments = len(route_nodes) - 1
        
        # Minimum 30 frames per second * 2 seconds minimum = 60 frames
        # Add 30 frames per segment
        min_frames = 60
        segment_frames = max(30, total_segments * 30)
        return max(min_frames, segment_frames)

    def create_animated_mp4(self, output_path: str, fps: int = 30, max_workers: int = 4):
        """Create an animated MP4 showing pulse effects on flows (parallelized frame rendering)."""
        print(f"Creating animated MP4: {output_path}")

        if cv2 is None:
            print("Error: OpenCV not installed. Install with: pip install opencv-python")
            return False

        # Calculate animation duration based on route length
        frames = self._calculate_animation_duration()
        print(f"  Animation duration: {frames} frames (~{frames/fps:.1f}s)")

        frame_paths = []
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Use ThreadPoolExecutor to render frames in parallel
            print(f"  Rendering with {max_workers} worker threads...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._render_frame, i, frames, temp_dir): i for i in range(frames)}
                completed_count = 0
                for completed in as_completed(futures):
                    try:
                        frame_num, frame_path = completed.result()
                        frame_paths.append((frame_num, frame_path))
                        completed_count += 1
                        if completed_count % 30 == 0:
                            print(f"  Generated {completed_count}/{frames} frames...")
                    except Exception as e:
                        print(f"Error rendering frame: {e}")

            # Sort frame paths by frame number to ensure correct order for MP4 encoding
            frame_paths.sort(key=lambda x: x[0])
            frame_paths = [path for _, path in frame_paths]

            # Encode frames to MP4
            print("Encoding MP4 video...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (1280, 720)
            )

            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)

            out.release()
            print(f"Successfully created: {output_path}")

            # Clean up temp frames
            for frame_path in frame_paths:
                Path(frame_path).unlink()
            temp_dir.rmdir()

            return True

        except Exception as e:
            print(f"Error creating MP4: {e}")
            return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Infrastructure Flow Animator (OpenCV + Plotly version)")
        print("Usage: python animate_flow_opencv.py <path_to_config.json>")
        print("Example: python animate_flow_opencv.py inputs/example_order_processing.json")
        print("\nThis version generates HTML and animated MP4 videos.")
        sys.exit(1)

    json_path = sys.argv[1]
    config = load_flow_config(json_path)

    visualizer = InfrastructureFlowVisualizer(config)

    # Create output directory
    output_dir = Path("media")
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    process_name_safe = config.get("process_name", "flow").replace(" ", "_").lower()
    output_path = output_dir / f"{process_name_safe}.mp4"

    print(f"Rendering: {config.get('process_name')}")
    visualizer.create_static_visualization(str(output_path))
    visualizer.create_animated_mp4(str(output_path))

    print(f"\nOutput saved to: {output_dir}/")


if __name__ == "__main__":
    main()
