"""Chainlit UI for the Proposal Agent System.

This app visualizes the multi-agent workflow using Chainlit's Step API.
Each agent (Planner, Researcher, Writer, Evaluator) helps build the final proposal.
"""

import os
import chainlit as cl
from langchain_core.messages import HumanMessage
from graph.graph import build_graph

# Initialize the graph once to ensure it's compiled
APP_GRAPH = build_graph()

# Directories for output
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    cl.user_session.set("graph", APP_GRAPH)
    cl.user_session.set("config", {"configurable": {"thread_id": cl.context.session.id}})
    cl.user_session.set("processed_ids", set())
    
    await cl.Message(
        content="**Welcome to the AI Proposal Agent!** 🚀\n\n"
                "I follow a multi-step process: Plan → Research → [Your Review] → Write → Critique → Output.\n\n"
                "**Enter your proposal request below:**"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user input, handle interrupts, and stream the graph execution."""
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")
    
    # Check current state to see if we are resuming from an interrupt
    state = graph.get_state(config)
    
    if state.next:
        # We are resuming after the researcher interrupt
        # The user's message is the feedback
        await graph.aupdate_state(config, {"user_feedback": message.content}, as_node="researcher")
        # Resume execution (passing None for inputs because we are resuming)
        stream = graph.astream_events(None, config, version="v2")
        await cl.Message(content=f"🔄 **Resuming with your feedback:** *\"{message.content}\"*").send()
    else:
        # Start fresh
        inputs = {
            "task": message.content,
            "messages": [HumanMessage(content=message.content)],
            "revision_count": 0,
            "score": 0.0,
            "user_feedback": ""
        }
        stream = graph.astream_events(inputs, config, version="v2")

    # Track active steps and main response
    active_steps = {}
    main_response = None
    MAJOR_NODES = {"planner", "researcher", "writer", "evaluator", "output"}

    async for event in stream:
        kind = event["event"]
        name = event["name"]
        data = event["data"]
        
        # --- Node Start ---
        if kind == "on_chain_start" and name in MAJOR_NODES:
            # Deduplicate steps based on run_id (to prevent re-showing steps on resume)
            run_id = event.get("run_id")
            processed_ids = cl.user_session.get("processed_ids")
            if processed_ids is None:
                processed_ids = set()
                cl.user_session.set("processed_ids", processed_ids)
            
            if run_id in processed_ids:
                continue
            processed_ids.add(run_id)

            # Steps are top-level timeline items (no parent_id) so they
            # render in correct chronological order alongside messages.
            step = cl.Step(name=name.capitalize(), type="run")
            await step.send()
            active_steps[name] = step
            
            # If it's the writer, prepare the main response message for streaming the proposal
            if name == "writer":
                main_response = cl.Message(content="")
                await main_response.send()
            
        # --- LLM Streaming ---
        elif kind == "on_chat_model_stream":
            content = data["chunk"].content
            if content:
                # 1. Stream to the active step
                for node_name in MAJOR_NODES:
                    if node_name in active_steps:
                        await active_steps[node_name].stream_token(content)
                        # 2. ALSO stream to main response ONLY if in 'writer' node
                        if node_name == "writer" and main_response:
                            await main_response.stream_token(content)
                        break

        # --- Node End ---
        elif kind == "on_chain_end" and name in MAJOR_NODES:
            if name in active_steps:
                step = active_steps[name]
                output = data.get("output")
                
                if name == "planner":
                    step.output = f"**Plan Generated**\n\n{output.get('plan', '')}"
                    await step.update()
                elif name == "researcher":
                    # After researcher finishes, the graph INTERRUPTS. 
                    # We need to explicitly ask the user to review.
                    res_data = output.get("research_data", "")
                    step.output = f"**Research Complete**\n\n{res_data}"
                    await step.update()
                    
                    await cl.Message(
                        content="✨ **Research Phase Complete.**\n\n"
                                "Please review the gathered data above. "
                                "You can **reply with any additional requirements** or just say **'Proceed'** to start writing the draft."
                    ).send()
                elif name == "writer":
                    step.output = "Draft generated (see main response)"
                    if main_response:
                        await main_response.update() # Finalize the stream
                    await step.update()
                elif name == "evaluator":
                    score = output.get("score")
                    critique = output.get("critique")
                    step.output = f"**Score: {score}/10**\n\n{critique}"
                    await step.update()
                elif name == "output":
                    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".md")]
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
                    if files:
                        latest_file = files[0]
                        abs_path = os.path.abspath(os.path.join(OUTPUT_DIR, latest_file))
                        step.output = f"Proposal saved to: {latest_file}"
                        await step.update()
                        
                        await cl.Message(
                            content=f"✅ **Proposal Ready for Download!**",
                            elements=[cl.File(name=latest_file, path=abs_path)]
                        ).send()
                    else:
                        await step.update()

                if name in active_steps:
                    del active_steps[name]
