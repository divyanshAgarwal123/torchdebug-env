"""
Gradio UI for the TorchDebug environment.

Provides an interactive browser-based interface for judges and developers
to manually step through debugging scenarios. Follows the REPL env's
gradio_builder pattern passed to create_app().
"""

from __future__ import annotations

import json
from typing import Any, Dict

try:
    import gradio as gr
except ImportError:
    gr = None  # Gradio optional — server works without it


def build_torchdebug_gradio_app(app: Any) -> Any:
    """Build a Gradio Blocks UI mounted into the OpenEnv app.

    This follows the reference pattern from repl_env/gradio_ui.py:
    the function receives the FastAPI app and returns a Gradio Blocks.
    """
    if gr is None:
        return None

    from server.torchdebug_environment import TorchDebugEnvironment
    from models import TorchDebugAction

    # Shared env instance for the Gradio session
    env = TorchDebugEnvironment()

    TASK_CHOICES = [
        ("🟢 Basic Failures (Easy)", "basic_failures"),
        ("🟡 Performance Issues (Medium)", "performance_issues"),
        ("🔴 Subtle Bugs (Hard)", "subtle_bugs"),
    ]

    ACTION_CHOICES = [
        "analyze_logs",
        "inspect_gradients",
        "inspect_data_pipeline",
        "inspect_model_architecture",
        "check_device_placement",
        "diagnose",
        "prescribe_fix",
        "request_hint",
    ]

    def do_reset(task_id: str) -> tuple:
        """Reset the environment with the selected task."""
        obs = env.reset(task_id=task_id)
        state_info = (
            f"**Task:** {obs.task_id}\n"
            f"**Difficulty:** {obs.difficulty}\n"
            f"**Step:** {obs.step_number}/{obs.max_steps}\n"
            f"**Reward:** {obs.reward:.2f}\n"
            f"**Done:** {obs.done}"
        )
        return (
            obs.task_description,
            obs.code_snippet or "N/A",
            obs.error_message or "No error — check logs for issues",
            obs.feedback,
            state_info,
            f"{obs.reward:.2f}",
            "",  # clear history
        )

    def do_step(action_type: str, diagnosis: str, fix_desc: str,
                fix_code: str, history: str) -> tuple:
        """Execute one step in the environment."""
        action = TorchDebugAction(
            action_type=action_type,
            diagnosis=diagnosis if action_type == "diagnose" else None,
            fix_description=fix_desc if action_type == "prescribe_fix" else None,
            fix_code=fix_code if action_type == "prescribe_fix" else None,
        )
        obs = env.step(action)

        state_info = (
            f"**Task:** {obs.task_id}\n"
            f"**Difficulty:** {obs.difficulty}\n"
            f"**Step:** {obs.step_number}/{obs.max_steps}\n"
            f"**Reward:** {obs.reward:.2f}\n"
            f"**Done:** {obs.done}\n"
            f"**Hints Used:** {obs.hints_used}"
        )

        # Append to history
        entry = f"**Step {obs.step_number}** | `{action_type}` → reward={obs.reward:.2f}"
        if obs.done:
            entry += " 🏁 **DONE**"
        new_history = f"{history}\n{entry}" if history else entry

        inspections = ""
        if obs.inspection_results:
            last = obs.inspection_results[-1]
            inspections = f"**{last.inspection_type}:**\n{last.findings}"

        return (
            obs.feedback,
            state_info,
            f"{obs.reward:.2f}",
            new_history,
            inspections,
        )

    with gr.Blocks(
        title="🔥 TorchDebug — Interactive Debugger",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🔥 TorchDebug — PyTorch Training Run Debugger\n"
            "An OpenEnv environment for AI agents to diagnose and fix "
            "real-world PyTorch training failures.\n\n"
            "**Select a task → Reset → Investigate → Diagnose → Fix**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎮 Controls")
                task_dd = gr.Dropdown(
                    choices=TASK_CHOICES,
                    value="basic_failures",
                    label="Task",
                )
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

                gr.Markdown("---")
                action_dd = gr.Dropdown(
                    choices=ACTION_CHOICES,
                    value="analyze_logs",
                    label="Action",
                )
                diagnosis_tb = gr.Textbox(
                    label="Diagnosis (for 'diagnose' action)",
                    placeholder="e.g. Learning rate too high causing NaN loss",
                    lines=2,
                )
                fix_desc_tb = gr.Textbox(
                    label="Fix Description (for 'prescribe_fix')",
                    placeholder="e.g. Reduce learning rate to 0.01",
                    lines=2,
                )
                fix_code_tb = gr.Textbox(
                    label="Fix Code (for 'prescribe_fix')",
                    placeholder="optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
                    lines=3,
                )
                step_btn = gr.Button("▶️ Step", variant="secondary")

                gr.Markdown("---")
                reward_display = gr.Textbox(label="Last Reward", interactive=False)
                state_display = gr.Markdown(label="State")

            with gr.Column(scale=2):
                gr.Markdown("### 📋 Scenario")
                desc_display = gr.Textbox(label="Description", lines=3, interactive=False)
                code_display = gr.Code(label="Code Snippet", language="python", interactive=False)
                error_display = gr.Textbox(label="Error Message", lines=2, interactive=False)

                gr.Markdown("### 💬 Feedback")
                feedback_display = gr.Textbox(label="Environment Feedback", lines=4, interactive=False)

                gr.Markdown("### 🔍 Inspection Results")
                inspection_display = gr.Markdown()

                gr.Markdown("### 📜 History")
                history_display = gr.Markdown()

        # Wire events
        reset_btn.click(
            do_reset,
            inputs=[task_dd],
            outputs=[desc_display, code_display, error_display,
                     feedback_display, state_display, reward_display,
                     history_display],
        )

        step_btn.click(
            do_step,
            inputs=[action_dd, diagnosis_tb, fix_desc_tb,
                    fix_code_tb, history_display],
            outputs=[feedback_display, state_display, reward_display,
                     history_display, inspection_display],
        )

    return demo
