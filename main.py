import asyncio
import json
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool

PROBLEM = "Solve Poverty"
MAX_ROUNDS = 3

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

# === Create Agents Dynamically ===
def create_agents(configs, problem):
    agents = []
    for cfg in configs:
        instructions = cfg["instructions"].replace("{problem}", problem)
        agent = Agent(name=cfg["name"], instructions=instructions)
        agents.append(agent)
    return agents

# === Main Execution ===
async def main():
    clarifying_questions = []
    agents = create_agents(agent_configs, PROBLEM)
    agent_map = {agent.name: agent for agent in agents}

    coordinator_agent = Agent(
        name="Coordinator Agent",
        instructions="You orchestrate a discussion among other agents to solve a complex problem. Use the tools provided to either ask specific agents follow-up questions or broadcast a question to all agents.",
        tools=[ask_specific_agent, ask_all_agents],
    )

    print("\n📚 Initial Analysis\n")
    initial_tasks = [Runner.run(agent, "Begin your analysis.") for agent in agents]
    outputs = await asyncio.gather(*initial_tasks)

    agent_outputs = {
        agent.name: output.final_output for agent, output in zip(agents, outputs)
    }

    for name, out in agent_outputs.items():
        print(f"--- {name} ---\n{out}\n")

    for round_num in range(MAX_ROUNDS):
        print(f"\n🌀 Panel Round {round_num + 1}\n")

        discussion = "\n\n".join([f"{k}:\n{v}" for k, v in agent_outputs.items()])

        coordinator_prompt = f"""
Current Discussion:
---
{discussion}
---

Based on the discussion, decide your next action:
1. If specific points need targeted follow-up, use the `ask_specific_agent` tool to generate questions for specific agents.
2. If a broader question would benefit all agents, use the `ask_all_agents` tool to generate a question for everyone.

Execute the chosen tool function.
"""
        try:
            coordinator_result = await Runner.run(coordinator_agent, coordinator_prompt)
            coordinator_output = coordinator_result.final_output.strip()

            print(f"🕵️ Coordinator output: {coordinator_output[:100]}...")

            parsed_questions = json.loads(coordinator_output)

            if isinstance(parsed_questions, list):
                print(f"🎯 Coordinator chose: Ask specific agents")
                round_tasks = []
                for item in parsed_questions:
                    agent_name = item.get("agent")
                    question = item.get("question")
                    if agent_name and question and agent_name in agent_map:
                        clarifying_questions.append(f"{agent_name}: {question}")
                        print(f"🧠 Coordinator asks {agent_name}: {question}\n")
                        round_tasks.append(Runner.run(agent_map[agent_name], question))
                    else:
                        print(f"⚠️ Invalid specific question item or unknown agent: {item}")
                if round_tasks:
                    specific_outputs = await asyncio.gather(*round_tasks)
                    idx = 0
                    for item in parsed_questions:
                        agent_name = item.get("agent")
                        if agent_name and item.get("question") and agent_name in agent_map:
                            agent_outputs[agent_name] = specific_outputs[idx].final_output
                            idx += 1
                else:
                    print("❓ No valid specific questions generated.")

            else:
                print(f"⚠️ Coordinator returned JSON but not a list: {parsed_questions}. Treating as broadcast.")
                raise ValueError("Expected JSON list from ask_specific_agent")

        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"📢 Coordinator chose: Broadcast to all agents")
            broadcast_question = coordinator_output
            clarifying_questions.append(f"All: {broadcast_question}")
            print(f"🗣️ Coordinator broadcasts: {broadcast_question}\n")

            broadcast_tasks = [Runner.run(agent, broadcast_question) for agent in agents]
            broadcast_outputs = await asyncio.gather(*broadcast_tasks)
            agent_outputs = {
                agent.name: output.final_output for agent, output in zip(agents, broadcast_outputs)
            }

        for name, out in agent_outputs.items():
            print(f"--- {name} ---\n{out}\n")

    markdown = f"""# AI Expert Panel Report: {PROBLEM}

## Final Summary of Agent Positions
"""
    for name, out in agent_outputs.items():
        markdown += f"\n### {name}\n\n{out}\n"

    markdown += "\n---\n\n## Discussion Questions Asked by Coordinator\n"
    markdown += "\n".join([f"- {q}" for q in clarifying_questions])

    print("\n📝 Final Markdown Report\n")
    print(markdown)

    report_filename = "panel_report.md"
    with open(report_filename, "w") as f:
        f.write(markdown)
    print(f"\n✅ Report saved to {report_filename}")

if __name__ == "__main__":
    asyncio.run(main())
