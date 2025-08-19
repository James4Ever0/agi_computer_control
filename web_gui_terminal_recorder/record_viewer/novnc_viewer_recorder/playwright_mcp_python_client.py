import asyncio

# reference readme: https://bgithub.xyz/microsoft/playwright-mcp

# install dependencies: pip install mcp (requiring python>=10)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import shlex


# Define server parameters
server_params = StdioServerParameters(
    command="docker",  # Command to run the server
    args=shlex.split(
        "run -i --rm --ipc=host mcp/playwright --browser chromium --caps pdf"
    ),  # Server script
)

# deprecated: mcp is for ai, not for programmer, since all tools are not type annotated in python

async def run_client():
    # Connect to the MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # Initialize the session

            # List available tools on the server
            tools = await session.list_tools()

            import rich

            rich.print("Available tools:", tools.tools)

            # # Call a tool (e.g., addition)
            # if any(tool.name == "add" for tool in tools.tools):
            #     result = await session.call_tool("add", {"a": 5, "b": 3})
            #     print("Addition result:", result.content[0].text)


# Run the client
if __name__ == "__main__":
    asyncio.run(run_client())
