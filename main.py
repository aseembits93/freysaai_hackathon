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

# === Agent Configuration (Pluggable) ===
agent_configs = [
    {
        "name": "Research Agent",
        "instructions": "You are the Research Agent. Summarize the issue: {problem}."
    },
    {
        "name": "Stakeholder Agent",
        "instructions": "You are the Stakeholder Agent. Map stakeholders and their iceberg motivations for: {problem}."
    },
    {
        "name": "Systems Agent",
        "instructions": "You are the Systems Thinking Agent. Analyze systemic loops in: {problem}."
    },
    {
        "name": "Futures Agent",
        "instructions": "You are the Futures Agent. Create a 2x2 scenario matrix and explain four future outcomes for: {problem}."
    },
    {
        "name": "Ethics Agent",
        "instructions": "You are the Ethics Agent. Consider fairness, blind spots, and values in: {problem}."
    },
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
async def run_panel_discussion(thread_id: str, problem: str, max_rounds: int = MAX_ROUNDS) -> None:
    """
    Runs the full panel discussion for a given problem and thread_id.
    Stores results in the database via the storage object.
    Returns None as results are stored, not returned directly.
    """
    # thread_id = str(uuid.uuid4()) # Removed: thread_id is now passed in
    print(f"\n🚀 Starting Background Panel Discussion for '{problem}' (Thread ID: {thread_id})\n")

    try: # Added top-level try/except for background task robustness
        clarifying_questions_log = []
        agents = create_agents(agent_configs, problem) # Use global agent_configs for now
        agent_map = {agent.name: agent for agent in agents}

        coordinator_agent = Agent(
            name="Coordinator Agent",
            instructions="You orchestrate a discussion among other agents to solve a complex problem...", # Keep instructions
            tools=[ask_specific_agent, ask_all_agents], # Assumes these tools use OPENAI_API_KEY env var
        )

        print("\n📚 Initial Analysis\n")
        initial_tasks = [
            run_agent_and_store(agent, "Begin your analysis.", thread_id)
            for agent in agents
        ]
        initial_outputs = await asyncio.gather(*initial_tasks)
        agent_outputs = {agent.name: output for agent, output in zip(agents, initial_outputs)}

        for name, out in agent_outputs.items():
            print(f"--- {name} (Initial) ---\n{out}\n")

        # Panel rounds
        for round_num in range(max_rounds):
            print(f"\n🌀 Panel Round {round_num + 1} (Thread: {thread_id})\n")

            # Construct discussion from the *latest* outputs stored
            discussion = "\n\n".join([f"{k}:\n{v}" for k, v in agent_outputs.items()])

            coordinator_prompt = f"""
Current Discussion (Round {round_num + 1}):
---
{discussion}
---
Based on the discussion, decide your next action... (keep prompt)
Execute the chosen tool function.
"""
            try:
                # Run coordinator - coordinator doesn't *need* to store its raw choice output, but its tool *calls* do
                # We need the *result* of the tool call (questions/question)
                print(f"🕵️ Running Coordinator for thread {thread_id}...")
                coordinator_result = await Runner.run(coordinator_agent, coordinator_prompt)
                coordinator_output = coordinator_result.final_output.strip()
                # We save the *interpreted* questions/action derived from the coordinator's output
                # We don't save the raw coordinator output itself unless desired (could add another save_message call)

                print(f"🕵️ Coordinator raw output: {coordinator_output[:100]}...")

                try:
                    parsed_questions = json.loads(coordinator_output)
                    if isinstance(parsed_questions, list):
                        print(f"🎯 Coordinator chose: Ask specific agents")
                        round_tasks = []
                        agents_to_run_this_round = {} # Track who is asked
                        for item in parsed_questions:
                            agent_name = item.get("agent")
                            question = item.get("question")
                            if agent_name and question and agent_name in agent_map:
                                log_entry = f"{agent_name}: {question}"
                                clarifying_questions_log.append(log_entry)
                                await storage.save_message(thread_id, "Coordinator", role="Coordinator", content=f"Ask {log_entry}") # Store coordinator action
                                print(f"🧠 Coordinator asks {agent_name}: {question}\n")
                                agents_to_run_this_round[agent_name] = question # Map agent to their question
                            else:
                                print(f"⚠️ Invalid specific question item or unknown agent: {item}")

                        if agents_to_run_this_round:
                             # Create tasks only for agents who were asked
                            round_tasks = [
                                run_agent_and_store(agent_map[name], question, thread_id)
                                for name, question in agents_to_run_this_round.items()
                            ]
                            specific_outputs = await asyncio.gather(*round_tasks)
                            # Update outputs dict with new results
                            idx = 0
                            for name in agents_to_run_this_round.keys():
                                 agent_outputs[name] = specific_outputs[idx] # Update with latest
                                 idx +=1
                        else:
                             print("❓ No valid specific questions generated.")
                    else:
                        print(f"⚠️ Coordinator returned JSON but not a list: {coordinator_output}. Treating as broadcast.")
                        raise ValueError("Expected JSON list from ask_specific_agent")

                except (json.JSONDecodeError, ValueError, TypeError):
                     # Treat as a broadcast question
                    print(f"📢 Coordinator chose: Broadcast to all agents")
                    broadcast_question = coordinator_output
                    clarifying_questions_log.append(f"All: {broadcast_question}")
                    await storage.save_message(thread_id, "Coordinator", role="Coordinator", content=f"Broadcast: {broadcast_question}") # Store coordinator action
                    print(f"🗣️ Coordinator broadcasts: {broadcast_question}\n")

                    broadcast_tasks = [
                        run_agent_and_store(agent, broadcast_question, thread_id)
                        for agent in agents
                    ]
                    broadcast_results = await asyncio.gather(*broadcast_tasks)
                    # Update all agent outputs
                    agent_outputs = {agent.name: result for agent, result in zip(agents, broadcast_results)}

            except Exception as e:
                 print(f"‼️ Error during coordinator round {round_num + 1}: {e}")
                 # Decide how to handle errors - continue, break, etc.
                 break # Example: Stop the discussion on error

            # Print results for the round
            for name, out in agent_outputs.items():
                print(f"--- {name} (Round {round_num + 1}) ---\n{out}\n")

        print(f"\n🏁 Panel Discussion Finished (Thread: {thread_id})\n")
        # Final outputs are the last recorded outputs for each agent
        # return thread_id, agent_outputs, clarifying_questions_log # Removed: Function returns None
    except Exception as e:
        # Catch errors during the overall process in the background task
        error_message = f"Unhandled error during background panel discussion for thread {thread_id}: {e}"
        print(f"�� {error_message}")
        try:
            # Attempt to log the error to the database
            await storage.save_message(thread_id, "System Error", role="System Error", content=error_message)
        except Exception as db_error:
            print(f"🚨🚨 Failed to log error to database: {db_error}")
        # Consider re-raising if FastAPI handles background task exceptions well,
        # otherwise, logging might be sufficient.


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
