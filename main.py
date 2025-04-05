import asyncio
import json
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool
import uuid
import time
import aiosqlite
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn # For running the app

PROBLEM = "Solve Poverty"
MAX_ROUNDS = 3

# === Database Setup ===
DB_FILE = "panel_discussions.db"

async def init_db():
    async with aiosqlite.connect(DB_FILE) as db:
        # Create table if it doesn't exist (for first run)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                thread_id TEXT NOT NULL,
                message_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        # Check if the 'role' column exists
        cursor = await db.execute("PRAGMA table_info(messages)")
        columns = [row[1] for row in await cursor.fetchall()]

        # Add the 'role' column if it doesn't exist (for migrations)
        if 'role' not in columns:
            print("Database schema migration: Adding 'role' column to messages table.")
            # Add column with a default value for existing rows
            await db.execute("ALTER TABLE messages ADD COLUMN role TEXT NOT NULL DEFAULT 'Agent'")

        # Ensure index exists
        await db.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON messages (thread_id)")
        await db.commit()

# === Storage Interface ===
class Message(BaseModel):
    thread_id: str
    message_id: str
    agent_name: str
    role: str
    content: str
    timestamp: float

class Storage:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path

    async def save_message(self, thread_id: str, agent_name: str, role: str, content: str) -> Message:
        message_id = str(uuid.uuid4())
        timestamp = time.time()
        message = Message(
            thread_id=thread_id,
            message_id=message_id,
            agent_name=agent_name,
            role=role,
            content=content,
            timestamp=timestamp
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (thread_id, message_id, agent_name, role, content, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (message.thread_id, message.message_id, message.agent_name, message.role, message.content, message.timestamp)
            )
            await db.commit()
        print(f"💾 Saved message {message_id} for agent {agent_name} (Role: {role}) in thread {thread_id}")
        return message

    async def get_messages_by_thread(self, thread_id: str) -> list[Message]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT thread_id, message_id, agent_name, role, content, timestamp FROM messages WHERE thread_id = ? ORDER BY timestamp ASC", (thread_id,)) as cursor:
                rows = await cursor.fetchall()
                return [Message(thread_id=row[0], message_id=row[1], agent_name=row[2], role=row[3], content=row[4], timestamp=row[5]) for row in rows]

# Initialize storage
storage = Storage()

agent_configs = [



  {


    "name": "Research Agent",


    "instructions": "You are the Research Agent. Your job is to conduct background research on the wicked problem: {problem}, using the uploaded documents and contextual materials. Start with framing using Rittel & Webber's criteria, and explore related cases or patterns. Be concise and structured. Highlight system-level insights from facts. Create a markdown summary with bullet points and references. Best practices: (1) Frame using Rittel & Webber's wicked problem criteria, (2) Include relevant historical or policy precedent (moonshots, past attempts), (3) Reference sources clearly, (4) Highlight system-level insights from facts."


  },


  {


    "name": "Stakeholder Agent",


    "instructions": "You are the Stakeholder Agent. Analyze and map stakeholders involved in the problem: {problem}. Use the Iceberg Model to show surface vs deep motivations, and describe their positions using a PESTLE framework. Show tradeoffs, power imbalances, and hidden tensions using structured tables or bullet lists. Create a markdown table and iceberg diagram description. Best practices: (1) Use PESTLE to map stakeholder goals, (2) Apply Iceberg model: Events → Patterns → Structures → Mental Models, (3) Highlight hidden tensions and alliances, (4) Show tradeoffs and power imbalances."


  },


  {


    "name": "Systems Agent",


    "instructions": "You are the Systems Thinking Agent. Identify causal feedback loops, system structures, and leverage points for the problem: {problem}. Use CLD-style text to explain loops. Highlight feedback delays and key bottlenecks. Summarize leverage points using bullet points or a table if applicable. Create a markdown loop table and leverage point description. Best practices: (1) Use CLDs (causal loop descriptions) with text if visuals unavailable, (2) Identify 2–3 leverage points, (3) Connect stakeholder tensions to systemic behaviors, (4) Highlight feedback delays, bottlenecks."


  },


  {


    "name": "Futures Agent",


    "instructions": "You are the Futures Agent. Use the Six Pillars of Futures Thinking to generate four scenarios for the problem: {problem} — one each for possible, plausible, probable, and preferred futures. Base these on a 2x2 matrix using two major uncertainties. Describe implications, risks, and opportunities in a tabular or bullet format. Create a scenario matrix and 4 descriptive summaries. Best practices: (1) Use uncertainties to build matrix (e.g., high tech vs low trust), (2) Name each scenario, (3) Describe implications, risks, and values, (4) Include brief summary table: Scenario | Traits | Opportunities | Risks."


  },


  {


    "name": "Experiential Agent",


    "instructions": "You are the Experiential Futures Agent. Translate scenarios related to the problem: {problem} into immersive, narrative experiences. Use the EXF Ladder: Setting → Scenario → Situation → Stuff. Include emotional tone, dilemmas, and Design Fiction elements. Describe any visuals (artifacts or vignettes) textually and clearly. Create a narrative and description table. Best practices: (1) Follow EXF Ladder: Setting → Scenario → Situation → Stuff, (2) Include emotional tone and real-world detail, (3) Use Design Fiction tools for provocation, (4) Summarize in table: Scenario | Emotion | Dilemma | Signal."


  },


  {


    "name": "Backcasting Agent",


    "instructions": "You are the Backcasting Agent. Starting from a preferred future scenario of the problem: {problem}, work backwards to map transitions. Break it down into Long-term, Mid-term, and Near-term phases. Identify bottlenecks, actors, and interventions. Use a table if helpful. Create a timeline narrative and table. Best practices: (1) Start with a clearly stated vision, (2) Work backwards in 3 phases: Long-term, Mid-term, Near-term, (3) Identify bottlenecks, decision points, (4) Summarize as: Stage | Intervention | Actor | Risk."


  },


  {


    "name": "Ethics Agent",


    "instructions": "You are the Ethics Agent. Review the outputs of other agents for the problem: {problem} and identify any bias, blind spots, or ethical tradeoffs. Provide improvements and highlight concerns, with specific notes per agent. Ensure vulnerable perspectives are considered. Create an ethical audit table and summary. Best practices: (1) Look for dominant perspectives that crowd out others, (2) Include empathy for vulnerable stakeholders, (3) Note inconsistencies or contradictions, (4) Provide improvements per agent or scenario."


  },


  {


    "name": "Synthesis Agent",


    "instructions": "You are the Synthesis Agent. Review and integrate the outputs of all other agents related to the problem: {problem}. Summarize the insights into a cohesive brief including: (1) Problem framing, (2) Stakeholder tensions, (3) System dynamics, (4) Scenarios, (5) Vision, (6) Action roadmap. Create a structured markdown with optional summary table. Best practices: (1) Refer to each agent's findings explicitly, (2) Keep conclusions concise, yet layered, (3) Surface contradictions and areas of consensus, (4) Offer final recommendation or strategic insight."


  }

]

# === Coordinator Tools ===
@function_tool
async def ask_specific_agent(discussion_summary: str) -> str:
    """
    Generate specific questions for specific agents based on the discussion.
    Returns a JSON array like: [{'agent': 'Agent Name', 'question': '...'}, ...]
    """
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a coordinator selecting multiple targeted questions for a multi-agent panel discussion."},
            {"role": "user", "content": f"""Given the discussion below, return a JSON array of follow-up questions like this: [{{"agent": "Agent Name", "question": "..."}}]

Discussion:
{discussion_summary}"""}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

@function_tool
async def ask_all_agents(discussion_summary: str) -> str:
    """
    Generate a single deep follow-up question to ask all agents based on the discussion.
    """
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a coordinator helping guide a multi-agent discussion."},
            {"role": "user", "content": f"Given the following panel discussion, generate a single deep follow-up question to ask all agents: {discussion_summary}"}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# === Modified Agent Runner ===
async def run_agent_and_store(agent: Agent, prompt: str, thread_id: str) -> str:
    """Runs an agent and stores its output."""
    print(f"🏃 Running agent {agent.name} for thread {thread_id}...")
    # Assuming Runner.run returns an object with a 'final_output' attribute
    result = await Runner.run(agent, prompt)
    output_content = result.final_output.strip()
    await storage.save_message(thread_id=thread_id, agent_name=agent.name, role="Agent", content=output_content)
    return output_content # Return the content for immediate use

# === Create Agents Dynamically ===
def create_agents(configs, problem):
    agents = []
    for cfg in configs:
        instructions = cfg["instructions"].replace("{problem}", problem)
        agent = Agent(name=cfg["name"], instructions=instructions)
        agents.append(agent)
    return agents

# === Modified Main Execution Logic (now callable) ===
# async def run_panel_discussion(thread_id: str, problem: str, max_rounds: int = MAX_ROUNDS) -> None:
#     """
#     Runs the full panel discussion for a given problem and thread_id.
#     Stores results in the database via the storage object.
#     Returns None as results are stored, not returned directly.
#     """
#     # thread_id = str(uuid.uuid4()) # Removed: thread_id is now passed in
#     print(f"\n🚀 Starting Background Panel Discussion for '{problem}' (Thread ID: {thread_id})\n")
#
#     try: # Added top-level try/except for background task robustness
#         clarifying_questions_log = []
#         agents = create_agents(agent_configs, problem) # Use global agent_configs for now
#         agent_map = {agent.name: agent for agent in agents}
#
#         coordinator_agent = Agent(
#             name="Coordinator Agent",
#             instructions="You orchestrate a discussion among other agents to solve a complex problem...", # Keep instructions
#             tools=[ask_specific_agent, ask_all_agents], # Assumes these tools use OPENAI_API_KEY env var
#         )
#
#         print("\n📚 Initial Analysis\n")
#         initial_tasks = [
#             run_agent_and_store(agent, "Begin your analysis.", thread_id)
#             for agent in agents
#         ]
#         initial_outputs = await asyncio.gather(*initial_tasks)
#         agent_outputs = {agent.name: output for agent, output in zip(agents, initial_outputs)}
#
#         for name, out in agent_outputs.items():
#             print(f"--- {name} (Initial) ---\n{out}\n")
#
#         # Panel rounds
#         for round_num in range(max_rounds):
#             print(f"\n🌀 Panel Round {round_num + 1} (Thread: {thread_id})\n")
#
#             # Construct discussion from the *latest* outputs stored
#             discussion = "\n\n".join([f"{k}:\n{v}" for k, v in agent_outputs.items()])
#
#             coordinator_prompt = f"""
# Current Discussion (Round {round_num + 1}):
# ---
# {discussion}
# ---
# Based on the discussion, decide your next action... (keep prompt)
# Execute the chosen tool function.
# """
#             try:
#                 # Run coordinator - coordinator doesn't *need* to store its raw choice output, but its tool *calls* do
#                 # We need the *result* of the tool call (questions/question)
#                 print(f"🕵️ Running Coordinator for thread {thread_id}...")
#                 coordinator_result = await Runner.run(coordinator_agent, coordinator_prompt)
#                 coordinator_output = coordinator_result.final_output.strip()
#                 # We save the *interpreted* questions/action derived from the coordinator's output
#                 # We don't save the raw coordinator output itself unless desired (could add another save_message call)
#
#                 print(f"🕵️ Coordinator raw output: {coordinator_output[:100]}...")
#
#                 try:
#                     parsed_questions = json.loads(coordinator_output)
#                     if isinstance(parsed_questions, list):
#                         print(f"🎯 Coordinator chose: Ask specific agents")
#                         round_tasks = []
#                         agents_to_run_this_round = {} # Track who is asked
#                         for item in parsed_questions:
#                             agent_name = item.get("agent")
#                             question = item.get("question")
#                             if agent_name and question and agent_name in agent_map:
#                                 log_entry = f"{agent_name}: {question}"
#                                 clarifying_questions_log.append(log_entry)
#                                 await storage.save_message(thread_id, "Coordinator", role="Coordinator", content=f"Ask {log_entry}") # Store coordinator action
#                                 print(f"🧠 Coordinator asks {agent_name}: {question}\n")
#                                 agents_to_run_this_round[agent_name] = question # Map agent to their question
#                             else:
#                                 print(f"⚠️ Invalid specific question item or unknown agent: {item}")
#
#                         if agents_to_run_this_round:
#                              # Create tasks only for agents who were asked
#                             round_tasks = [
#                                 run_agent_and_store(agent_map[name], question, thread_id)
#                                 for name, question in agents_to_run_this_round.items()
#                             ]
#                             specific_outputs = await asyncio.gather(*round_tasks)
#                             # Update outputs dict with new results
#                             idx = 0
#                             for name in agents_to_run_this_round.keys():
#                                  agent_outputs[name] = specific_outputs[idx] # Update with latest
#                                  idx +=1
#                         else:
#                              print("❓ No valid specific questions generated.")
#                     else:
#                         print(f"⚠️ Coordinator returned JSON but not a list: {coordinator_output}. Treating as broadcast.")
#                         raise ValueError("Expected JSON list from ask_specific_agent")
#
#                 except (json.JSONDecodeError, ValueError, TypeError):
#                      # Treat as a broadcast question
#                     print(f"📢 Coordinator chose: Broadcast to all agents")
#                     broadcast_question = coordinator_output
#                     clarifying_questions_log.append(f"All: {broadcast_question}")
#                     await storage.save_message(thread_id, "Coordinator", role="Coordinator", content=f"Broadcast: {broadcast_question}") # Store coordinator action
#                     print(f"🗣️ Coordinator broadcasts: {broadcast_question}\n")
#
#                     broadcast_tasks = [
#                         run_agent_and_store(agent, broadcast_question, thread_id)
#                         for agent in agents
#                     ]
#                     broadcast_results = await asyncio.gather(*broadcast_tasks)
#                     # Update all agent outputs
#                     agent_outputs = {agent.name: result for agent, result in zip(agents, broadcast_results)}
#
#             except Exception as e:
#                  print(f"‼️ Error during coordinator round {round_num + 1}: {e}")
#                  # Decide how to handle errors - continue, break, etc.
#                  break # Example: Stop the discussion on error
#
#             # Print results for the round
#             for name, out in agent_outputs.items():
#                 print(f"--- {name} (Round {round_num + 1}) ---\n{out}\n")
#
#         print(f"\n🏁 Panel Discussion Finished (Thread: {thread_id})\n")
#         # Final outputs are the last recorded outputs for each agent
#         # return thread_id, agent_outputs, clarifying_questions_log # Removed: Function returns None
#     except Exception as e:
#         # Catch errors during the overall process in the background task
#         error_message = f"Unhandled error during background panel discussion for thread {thread_id}: {e}"
#         print(f"�� {error_message}")
#         try:
#             # Attempt to log the error to the database
#             await storage.save_message(thread_id, "System Error", role="System Error", content=error_message)
#         except Exception as db_error:
#             print(f"🚨🚨 Failed to log error to database: {db_error}")
#         # Consider re-raising if FastAPI handles background task exceptions well,
#         # otherwise, logging might be sufficient.

async def run_panel_discussion(thread_id: str, problem: str, max_rounds: int = MAX_ROUNDS) -> None:
    """
    Runs a sophisticated panel discussion for a given problem and thread_id.

    Features:
    - Dynamic discussion flow based on agent interactions
    - Memory of key insights across rounds
    - Explicit debate and consensus-building phases
    - Automatic detection of emerging themes
    - Cross-pollination of ideas between agents

    Args:
        thread_id: Unique identifier for this discussion
        problem: The problem statement to analyze
        max_rounds: Maximum number of discussion rounds

    Returns:
        None (results are stored in the database)
    """
    print(f"\n🚀 Starting Background Panel Discussion for '{problem}' (Thread ID: {thread_id})\n")

    try:
        # Track discussion state across rounds
        discussion_state = {
            "key_insights": [],
            "open_questions": [],
            "emerging_themes": set(),
            "disagreements": [],
            "consensus_points": [],
            "round_summaries": []
        }

        agents = create_agents(agent_configs, problem)
        agent_map = {agent.name: agent for agent in agents}

        # Enhanced coordinator with additional tools
        coordinator_agent = Agent(
            name="Coordinator Agent",
            instructions="""You orchestrate a sophisticated discussion among expert agents to solve a complex problem.
            Your role is to:
            1. Identify patterns, contradictions, and insights across agent contributions
            2. Facilitate productive debate when agents have different perspectives
            3. Guide the discussion toward deeper analysis and practical solutions
            4. Ensure diverse perspectives are heard and integrated
            5. Track emerging themes and ensure the discussion evolves organically

            Track the discussion state carefully, and choose tools that will create the most generative dialogue.""",
            tools=[ask_specific_agent, ask_all_agents],
        )

        # Save the problem statement to the thread
        await storage.save_message(thread_id=thread_id, agent_name="System", role="Problem", content=problem)

        print("\n📚 Initial Analysis\n")
        # Enhanced initial prompts that establish agent expertise and approach
        initial_prompts = {
            "Research Agent": "Begin your analysis by examining the core aspects of this problem, key statistics, and fundamental causes. Focus on establishing a solid factual foundation for our discussion.",
            "Stakeholder Agent": "Map the ecosystem of stakeholders affected by this problem. Identify their explicit needs and implicit motivations. Consider power dynamics and whose voices might be missing.",
            "Systems Agent": "Analyze the interconnected systems, feedback loops, and leverage points relevant to this problem. What are the structural patterns that perpetuate this issue?",
            "Futures Agent": "Consider multiple possible futures related to this problem. What emerging trends or weak signals might significantly impact how this problem evolves?",
            "Ethics Agent": "Examine the ethical dimensions, values conflicts, and hidden assumptions embedded in how we frame this problem. What perspectives or considerations are we overlooking?"
        }

        # Run initial analysis with tailored prompts
        initial_tasks = [
            run_agent_and_store(agent, initial_prompts.get(agent.name, "Begin your analysis."), thread_id)
            for agent in agents
        ]
        initial_outputs = await asyncio.gather(*initial_tasks)
        agent_outputs = {agent.name: output for agent, output in zip(agents, initial_outputs)}

        for name, out in agent_outputs.items():
            print(f"--- {name} (Initial) ---\n{out}\n")

        # Extract and save key insights from initial analysis
        await extract_and_save_insights(agent_outputs, thread_id, discussion_state)

        # Main discussion rounds - now with adaptive flow and more sophisticated coordination
        for round_num in range(max_rounds):
            print(f"\n🌀 Panel Round {round_num + 1} (Thread: {thread_id})\n")

            # Determine discussion phase based on the round number and state
            phase = determine_discussion_phase(round_num, max_rounds, discussion_state)
            await storage.save_message(thread_id, "System", role="Phase",
                                       content=f"Round {round_num+1} Phase: {phase}")

            # Construct enhanced discussion context with more structure
            discussion = construct_enhanced_discussion_context(
                agent_outputs,
                discussion_state,
                round_num,
                phase
            )

            # Run coordinator with phase-appropriate prompting
            coordinator_prompt = f"""
Current Discussion (Round {round_num + 1}, Phase: {phase}):
---
{discussion}
---

Discussion State:
- Key Insights: {', '.join(discussion_state['key_insights'][:3])}
- Open Questions: {', '.join(discussion_state['open_questions'][:3])}
- Emerging Themes: {', '.join(list(discussion_state['emerging_themes'])[:3])}
- Disagreements: {', '.join(discussion_state['disagreements'][:2])}
- Consensus Points: {', '.join(discussion_state['consensus_points'][:2])}

Based on the current discussion phase ({phase}) and state, decide your next action:
1. If you need deeper exploration from specific agents, use ask_specific_agent
2. If you want to guide the whole panel, use ask_all_agents

Execute the chosen tool function with a question that advances the discussion.
"""
            try:
                print(f"🕵️ Running Coordinator for thread {thread_id} (Round {round_num+1}, Phase: {phase})...")
                coordinator_result = await Runner.run(coordinator_agent, coordinator_prompt)
                coordinator_output = coordinator_result.final_output.strip()

                # Process coordinator decision with enhanced context
                agent_outputs = await process_coordinator_decision(
                    coordinator_output,
                    agent_map,
                    agent_outputs,
                    thread_id,
                    round_num,
                    phase,
                    discussion_state
                )

                # Update discussion state based on new outputs
                await update_discussion_state(agent_outputs, thread_id, discussion_state, round_num)

                # Generate round summary if appropriate
                if should_generate_summary(round_num, max_rounds, phase):
                    summary = await generate_round_summary(agent_outputs, discussion_state, round_num, thread_id)
                    discussion_state["round_summaries"].append(summary)

            except Exception as e:
                print(f"‼️ Error during coordinator round {round_num + 1}: {e}")
                # Log error but continue to next round if possible
                await storage.save_message(thread_id, "System Error", role="Error",
                                           content=f"Error in round {round_num+1}: {str(e)}")
                continue  # Try to continue rather than break

            # Print results for the round
            for name, out in agent_outputs.items():
                print(f"--- {name} (Round {round_num + 1}) ---\n{out}\n")

        # Generate final synthesis
        await generate_final_synthesis(agent_outputs, discussion_state, thread_id)

        print(f"\n🏁 Panel Discussion Finished (Thread: {thread_id})\n")

    except Exception as e:
        error_message = f"Unhandled error during background panel discussion for thread {thread_id}: {e}"
        print(f"🚨 {error_message}")
        try:
            await storage.save_message(thread_id, "System Error", role="System Error", content=error_message)
        except Exception as db_error:
            print(f"🚨🚨 Failed to log error to database: {db_error}")


# Helper functions for enhanced discussion

async def extract_and_save_insights(agent_outputs, thread_id, discussion_state):
    """Extract key insights from initial agent outputs and update discussion state."""
    # In a real implementation, you might use an LLM to extract these insights
    insights_text = "\n\n".join(agent_outputs.values())
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Extract key insights and open questions from this multi-agent discussion."},
            {"role": "user", "content": f"Analyze these outputs and identify: 1) Key insights, 2) Open questions, 3) Emerging themes:\n\n{insights_text}"}
        ],
        temperature=0.7,
    )

    analysis = response.choices[0].message.content.strip()
    await storage.save_message(thread_id, "System", role="Analysis", content=analysis)

    # Parse the response to extract structured information (simplified parsing)
    if "Key insights:" in analysis:
        insights_section = analysis.split("Key insights:")[1].split("Open questions:")[0]
        insights = [i.strip() for i in insights_section.split("\n") if i.strip() and i.strip().startswith("-")]
        discussion_state["key_insights"].extend([i.lstrip("- ") for i in insights])

    if "Open questions:" in analysis:
        questions_section = analysis.split("Open questions:")[1].split("Emerging themes:")[0] if "Emerging themes:" in analysis else analysis.split("Open questions:")[1]
        questions = [q.strip() for q in questions_section.split("\n") if q.strip() and q.strip().startswith("-")]
        discussion_state["open_questions"].extend([q.lstrip("- ") for q in questions])

    if "Emerging themes:" in analysis:
        themes_section = analysis.split("Emerging themes:")[1]
        themes = [t.strip() for t in themes_section.split("\n") if t.strip() and t.strip().startswith("-")]
        discussion_state["emerging_themes"].update([t.lstrip("- ") for t in themes])


def determine_discussion_phase(round_num, max_rounds, discussion_state):
    """Determine the appropriate discussion phase based on progress."""
    if round_num == 0:
        return "Problem Definition"
    elif round_num < max_rounds // 3:
        return "Exploration"
    elif round_num < (max_rounds * 2) // 3:
        return "Analysis and Debate"
    elif round_num < max_rounds - 1:
        return "Solution Development"
    else:
        return "Synthesis and Integration"


def construct_enhanced_discussion_context(agent_outputs, discussion_state, round_num, phase):
    """Create a rich discussion context with state information."""
    # Recent agent outputs
    recent_outputs = "\n\n".join([f"{k}:\n{v}" for k, v in agent_outputs.items()])

    # Add discussion state elements based on the phase
    if phase == "Problem Definition":
        context = f"PHASE: PROBLEM DEFINITION\n\n{recent_outputs}"
    elif phase == "Exploration":
        # Include key insights and open questions
        insights = "\n".join(discussion_state["key_insights"][:5])
        questions = "\n".join(discussion_state["open_questions"][:5])
        context = f"PHASE: EXPLORATION\n\nKey Insights So Far:\n{insights}\n\nOpen Questions:\n{questions}\n\nRecent Contributions:\n{recent_outputs}"
    elif phase == "Analysis and Debate":
        # Highlight disagreements and themes
        disagreements = "\n".join(discussion_state["disagreements"])
        themes = "\n".join(list(discussion_state["emerging_themes"]))
        context = f"PHASE: ANALYSIS AND DEBATE\n\nEmerging Themes:\n{themes}\n\nKey Disagreements:\n{disagreements}\n\nRecent Contributions:\n{recent_outputs}"
    elif phase == "Solution Development":
        # Focus on building consensus and solutions
        consensus = "\n".join(discussion_state["consensus_points"])
        context = f"PHASE: SOLUTION DEVELOPMENT\n\nEstablished Consensus:\n{consensus}\n\nRecent Contributions:\n{recent_outputs}"
    else:  # Synthesis
        # Bring everything together
        insights = "\n".join(discussion_state["key_insights"])
        consensus = "\n".join(discussion_state["consensus_points"])
        context = f"PHASE: SYNTHESIS\n\nKey Insights:\n{insights}\n\nConsensus Points:\n{consensus}\n\nFinal Perspectives:\n{recent_outputs}"

    return context


async def process_coordinator_decision(coordinator_output, agent_map, agent_outputs, thread_id, round_num, phase, discussion_state):
    """Process the coordinator's decision with enhanced context awareness."""
    try:
        parsed_questions = json.loads(coordinator_output)
        if isinstance(parsed_questions, list):
            print(f"🎯 Coordinator chose: Ask specific agents")

            # Add phase-specific context to questions
            enhanced_questions = []
            agents_to_run_this_round = {}

            for item in parsed_questions:
                agent_name = item.get("agent")
                question = item.get("question")

                if agent_name and question and agent_name in agent_map:
                    # Enhance question with phase context and relevant state information
                    enhanced_question = enhance_question_with_context(
                        question, agent_name, phase, discussion_state, round_num
                    )

                    log_entry = f"{agent_name}: {enhanced_question}"
                    await storage.save_message(thread_id, "Coordinator", role="Coordinator",
                                               content=f"Ask {agent_name}: {enhanced_question}")
                    print(f"🧠 Coordinator asks {agent_name}: {enhanced_question}\n")
                    agents_to_run_this_round[agent_name] = enhanced_question
                else:
                    print(f"⚠️ Invalid specific question item or unknown agent: {item}")

            if agents_to_run_this_round:
                round_tasks = [
                    run_agent_and_store(agent_map[name], question, thread_id)
                    for name, question in agents_to_run_this_round.items()
                ]
                specific_outputs = await asyncio.gather(*round_tasks)

                # Update outputs dict with new results
                idx = 0
                for name in agents_to_run_this_round.keys():
                    agent_outputs[name] = specific_outputs[idx]
                    idx += 1

                # For agents not questioned this round, provide them with a summary
                # of what was discussed to keep them in the loop
                if len(agents_to_run_this_round) < len(agent_map) and round_num > 0:
                    await inform_inactive_agents(
                        agent_map,
                        agents_to_run_this_round.keys(),
                        specific_outputs,
                        thread_id
                    )
            else:
                print("❓ No valid specific questions generated.")
        else:
            raise ValueError("Expected JSON list from ask_specific_agent")

    except (json.JSONDecodeError, ValueError, TypeError):
        # Handle as broadcast question with phase-specific enhancement
        print(f"📢 Coordinator chose: Broadcast to all agents")
        broadcast_question = coordinator_output

        # Enhance the broadcast question for the current phase
        enhanced_broadcast = enhance_broadcast_with_context(
            broadcast_question, phase, discussion_state, round_num
        )

        await storage.save_message(thread_id, "Coordinator", role="Coordinator",
                                   content=f"Broadcast: {enhanced_broadcast}")
        print(f"🗣️ Coordinator broadcasts: {enhanced_broadcast}\n")

        broadcast_tasks = [
            run_agent_and_store(agent, enhanced_broadcast, thread_id)
            for agent in agent_map.values()
        ]
        broadcast_results = await asyncio.gather(*broadcast_tasks)

        # Update all agent outputs
        agent_outputs = {agent.name: result for agent, result in zip(agent_map.values(), broadcast_results)}

    return agent_outputs


def enhance_question_with_context(question, agent_name, phase, discussion_state, round_num):
    """Add rich context to a specific agent question based on the discussion phase."""
    # Base context from the phase
    phase_context = {
        "Problem Definition": "As we're defining the problem scope, ",
        "Exploration": "In our exploration phase, ",
        "Analysis and Debate": "As we analyze differing perspectives, ",
        "Solution Development": "As we develop practical solutions, ",
        "Synthesis and Integration": "As we synthesize our conclusions, "
    }.get(phase, "")

    # Agent-specific context based on their role
    agent_context = {
        "Research Agent": "Considering the factual evidence, ",
        "Stakeholder Agent": "From the perspective of key stakeholders, ",
        "Systems Agent": "Looking at the systemic interactions, ",
        "Futures Agent": "Considering future scenarios, ",
        "Ethics Agent": "From an ethical standpoint, "
    }.get(agent_name, "")

    # Add relevant discussion state elements
    state_context = ""
    if discussion_state["key_insights"] and round_num > 0:
        relevant_insight = random.choice(discussion_state["key_insights"])
        state_context += f"Building on the insight that '{relevant_insight}', "

    if agent_name in discussion_state.get("agent_contributions", {}) and round_num > 1:
        state_context += f"Following up on your earlier point about {discussion_state['agent_contributions'].get(agent_name, '')}, "

    # Combine all context elements with the original question
    enhanced_question = f"{phase_context}{agent_context}{state_context}{question}"

    return enhanced_question


def enhance_broadcast_with_context(question, phase, discussion_state, round_num):
    """Enhance a broadcast question with phase-appropriate context."""
    # Add phase-specific framing
    phase_prefix = {
        "Problem Definition": "As we establish our shared understanding of this problem, ",
        "Exploration": "To explore the dimensions of this challenge more deeply, ",
        "Analysis and Debate": "To examine tensions and contradictions in our analysis, ",
        "Solution Development": "To develop more integrated and effective solutions, ",
        "Synthesis and Integration": "To synthesize our collective insights, "
    }.get(phase, "")

    # Add discussion state context
    state_context = ""
    if discussion_state["emerging_themes"] and len(discussion_state["emerging_themes"]) > 0:
        themes = list(discussion_state["emerging_themes"])
        state_context = f"Considering the emerging themes of {', '.join(themes[:2])}, "

    # Combine with original question
    enhanced_question = f"{phase_prefix}{state_context}{question}"

    return enhanced_question


async def inform_inactive_agents(agent_map, active_agents, recent_outputs, thread_id):
    """Keep inactive agents informed of the discussion progress."""
    # Create a summary of what active agents discussed
    summary = "\n".join([f"{agent}: {output[:100]}..." for agent, output in zip(active_agents, recent_outputs)])

    # Find agents that weren't active this round
    inactive_agents = [name for name in agent_map.keys() if name not in active_agents]

    if inactive_agents:
        update_message = f"While other agents were discussing, here's a brief update on their perspectives: {summary}"

        # Log this update but don't run the agents - just keep them informed for continuity
        for name in inactive_agents:
            await storage.save_message(thread_id, "System", role="Update",
                                       content=f"Update for {name}: {update_message}")


async def update_discussion_state(agent_outputs, thread_id, discussion_state, round_num):
    """Update the discussion state based on the latest agent outputs."""
    # Combine all agent outputs for analysis
    combined_output = "\n\n".join([f"{name}: {output}" for name, output in agent_outputs.items()])

    # Use OpenAI to analyze the outputs
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Analyze this discussion round and extract key elements."},
            {"role": "user", "content": f"""Analyze the latest round of discussion and identify:
1. New key insights
2. New open questions
3. Emerging themes
4. Areas of disagreement
5. Points of consensus

Discussion:
{combined_output}"""}
        ],
        temperature=0.7,
    )

    analysis = response.choices[0].message.content.strip()

    # Save the analysis
    await storage.save_message(thread_id, "System", role="RoundAnalysis",
                               content=f"Round {round_num+1} Analysis: {analysis}")

    # Update discussion state based on analysis (simplified parsing)
    if "Key insights:" in analysis:
        insights_section = analysis.split("Key insights:")[1].split("Open questions:")[0]
        new_insights = [i.strip() for i in insights_section.split("\n") if i.strip() and i.strip().startswith("-")]
        discussion_state["key_insights"].extend([i.lstrip("- ") for i in new_insights])

    if "Open questions:" in analysis:
        questions_section = analysis.split("Open questions:")[1].split("Emerging themes:")[0]
        new_questions = [q.strip() for q in questions_section.split("\n") if q.strip() and q.strip().startswith("-")]
        discussion_state["open_questions"].extend([q.lstrip("- ") for q in new_questions])

    if "Emerging themes:" in analysis:
        themes_section = analysis.split("Emerging themes:")[1].split("Areas of disagreement:")[0]
        new_themes = [t.strip() for t in themes_section.split("\n") if t.strip() and t.strip().startswith("-")]
        discussion_state["emerging_themes"].update([t.lstrip("- ") for t in new_themes])

    if "Areas of disagreement:" in analysis:
        disagreement_section = analysis.split("Areas of disagreement:")[1].split("Points of consensus:")[0]
        new_disagreements = [d.strip() for d in disagreement_section.split("\n") if d.strip() and d.strip().startswith("-")]
        discussion_state["disagreements"] = [d.lstrip("- ") for d in new_disagreements]  # Replace with new

    if "Points of consensus:" in analysis:
        consensus_section = analysis.split("Points of consensus:")[1]
        new_consensus = [c.strip() for c in consensus_section.split("\n") if c.strip() and c.strip().startswith("-")]
        discussion_state["consensus_points"].extend([c.lstrip("- ") for c in new_consensus])

    # Track agent-specific contributions (simplified)
    discussion_state["agent_contributions"] = discussion_state.get("agent_contributions", {})
    for name, output in agent_outputs.items():
        # Extract a key contribution (simplified - in real implementation, use an LLM)
        first_sentence = output.split(".")[0]
        if len(first_sentence) > 10:
            discussion_state["agent_contributions"][name] = first_sentence


def should_generate_summary(round_num, max_rounds, phase):
    """Determine if we should generate a summary at this point."""
    # Generate summary at phase transitions or every 2 rounds
    return phase in ["Solution Development", "Synthesis and Integration"] or round_num % 2 == 1


async def generate_round_summary(agent_outputs, discussion_state, round_num, thread_id):
    """Generate a summary of the current round."""
    # Combine recent outputs and state information
    combined_data = "\n\n".join([f"{name}: {output}" for name, output in agent_outputs.items()])
    state_summary = f"""
Key Insights: {', '.join(discussion_state['key_insights'][-5:])}
Open Questions: {', '.join(discussion_state['open_questions'][-3:])}
Emerging Themes: {', '.join(list(discussion_state['emerging_themes'])[-3:])}
Disagreements: {', '.join(discussion_state['disagreements'][:2])}
Consensus Points: {', '.join(discussion_state['consensus_points'][-3:])}
"""

    # Generate summary
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Create a concise, insightful summary of this discussion round."},
            {"role": "user", "content": f"""Create a summary of Round {round_num+1} that captures:
1. Key developments and insights
2. Shifts in understanding
3. Emerging questions
4. Next steps

Discussion data:
{combined_data}

Discussion state:
{state_summary}"""}
        ],
        temperature=0.7,
    )

    summary = response.choices[0].message.content.strip()

    # Save the summary
    await storage.save_message(thread_id, "System", role="RoundSummary",
                               content=f"Round {round_num+1} Summary: {summary}")

    return summary


async def generate_final_synthesis(agent_outputs, discussion_state, thread_id):
    """Generate a final synthesis of the entire discussion."""
    # Combine all key elements from the discussion state
    insights = "\n".join(discussion_state["key_insights"])
    themes = "\n".join(list(discussion_state["emerging_themes"]))
    consensus = "\n".join(discussion_state["consensus_points"])
    summaries = "\n\n".join(discussion_state["round_summaries"])

    synthesis_data = f"""
KEY INSIGHTS:
{insights}

EMERGING THEMES:
{themes}

CONSENSUS POINTS:
{consensus}

ROUND SUMMARIES:
{summaries}

FINAL AGENT PERSPECTIVES:
{"".join([f"\n\n{name}:\n{output}" for name, output in agent_outputs.items()])}
"""

    # Generate synthesis
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Create a comprehensive synthesis of this multi-agent panel discussion."},
            {"role": "user", "content": f"""Create a final synthesis that:
1. Summarizes the key insights and developments
2. Identifies the most important themes and patterns
3. Presents an integrated understanding of the problem
4. Outlines the most promising solution directions
5. Highlights remaining open questions and areas for further exploration

Discussion data:
{synthesis_data}"""}
        ],
        temperature=0.7,
    )

    synthesis = response.choices[0].message.content.strip()

    # Save the synthesis
    await storage.save_message(thread_id, "System", role="FinalSynthesis", content=synthesis)

    return synthesis


# === FastAPI App Setup ===
app = FastAPI(title="AI Expert Panel API")

@app.on_event("startup")
async def startup_event():
    await init_db()
    print("Database initialized.")

class RunPanelRequest(BaseModel):
    problem: str = PROBLEM # Default to global PROBLEM
    max_rounds: int = MAX_ROUNDS # Default to global MAX_ROUNDS

class RunPanelResponse(BaseModel):
    thread_id: str
    final_outputs: dict[str, str]
    questions_asked: list[str]
    report_preview: str # Add a preview of the final report

# Define a response model for starting the panel
class StartPanelResponse(BaseModel):
    thread_id: str
    status: str

# The endpoint uses BackgroundTasks
@app.post("/run_panel", response_model=StartPanelResponse, status_code=202) # Use 202 Accepted
async def api_start_panel(request: RunPanelRequest, background_tasks: BackgroundTasks):
    """
    Starts a new AI expert panel discussion in the background.
    Returns the thread_id immediately. Check the /thread/{thread_id}
    endpoint to retrieve messages as they are generated.
    """
    # 1. Generate ID immediately
    thread_id = str(uuid.uuid4())
    print(f"Received request to start panel for '{request.problem}'. Assigning Thread ID: {thread_id}")

    # 2. Schedule the function to run in the background, passing the thread_id
    background_tasks.add_task(
        run_panel_discussion,  # The actual work function
        thread_id=thread_id,   # Pass the generated ID here
        problem=request.problem,
        max_rounds=request.max_rounds
    )

    # 3. Return the ID immediately (non-blocking)
    return StartPanelResponse(
        thread_id=thread_id,
        status="Panel discussion started in background."
    )

@app.get("/thread/{thread_id}", response_model=list[Message])
async def get_thread_messages(thread_id: str):
    """
    Retrieves all messages associated with a specific discussion thread_id,
    ordered by timestamp.
    """
    messages = await storage.get_messages_by_thread(thread_id)
    if not messages:
        raise HTTPException(status_code=404, detail=f"Thread ID '{thread_id}' not found.")
    return messages


# === Main block to run FastAPI ===
# Remove the old if __name__ == "__main__": asyncio.run(main()) block

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Ensure OPENAI_API_KEY environment variable is set.")
    # Make sure init_db() runs before server starts fully accepting requests
    # Uvicorn handles the asyncio loop setup when run this way
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # Note: reload=True is good for development, remove for production
