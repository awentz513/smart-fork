# Smart Fork VS Code Extension

AI-powered fork detection for Claude Code sessions. Find and reuse relevant past work directly from VS Code.

## Features

### Search Sessions
Find relevant past Claude Code sessions based on what you're working on:
- Search by keywords, technologies, or problem types
- View formatted search results in a dedicated panel
- See session previews, dates, and relevance scores

### Fork from Sessions
Fork from a past session to continue where you left off:
- View session preview before forking
- One-click copy of fork command
- Seamless integration with Claude Code

### View Fork History
See your fork history and track session relationships:
- Chronological fork history
- Parent-child session relationships
- Quick access to past forks

## Requirements

- VS Code 1.80.0 or higher
- Python 3.9 or higher
- Smart Fork MCP server installed ( from the Smart Fork directory)
- Claude Code sessions directory at 

## Installation

### From Source

1. Clone the Smart Fork repository
2. Navigate to the extension directory:
   
3. Install dependencies:
   
4. Compile the extension:
   
5. Open the extension directory in VS Code
6. Press F5 to launch the extension in debug mode

### Configuration

The extension can be configured via VS Code settings:

- : Path to Python executable (default: )
- : Path to MCP server module (default: )
- : Auto-start MCP server on activation (default: )
- : Max search results to display (default: )

## Usage

### Search for Sessions

1. Open the Command Palette ( on Mac,  on Windows/Linux)
2. Type 
3. Enter your search query (e.g., "authentication API", "React component")
4. View results in the Smart Fork panel

### Fork from a Session

1. Open the Command Palette
2. Type 
3. Enter the session ID from your search results
4. Review the session preview
5. Click "Fork" to copy the fork command
6. Run the command in Claude Code

### View Fork History

1. Open the Command Palette
2. Type 
3. View your fork history in a new document

## Commands

- : Search for relevant sessions
- : Fork from a specific session
- : View your fork history

## Troubleshooting

### MCP Server Connection Issues

If the extension can't connect to the MCP server:

1. Verify Smart Fork is installed: 
2. Check Python path in settings matches your environment
3. Ensure the MCP server can start manually: 
4. Check the Output panel (View > Output) and select "Smart Fork" for error messages

### No Results Found

If searches return no results:

1. Verify the database is initialized: Run 
2. Check that Claude Code sessions exist in 
3. Try broader search terms
4. Check the session registry: 

### Extension Won't Activate

1. Check VS Code version is 1.80.0 or higher
2. Verify TypeScript compilation completed successfully
3. Check the VS Code Extension Host output for errors
4. Try reloading the window ( or )

## Development

### Building



### Watching for Changes



### Debugging

1. Open the extension directory in VS Code
2. Press F5 to launch Extension Development Host
3. Set breakpoints in TypeScript files
4. Test commands in the development instance

## Architecture

The extension consists of three main components:

1. **extension.ts**: Main entry point, registers commands and manages lifecycle
2. **mcpClient.ts**: MCP client for communicating with the Smart Fork server
3. **searchResultsPanel.ts**: Webview panel for displaying search results

The extension communicates with the Smart Fork MCP server via JSON-RPC over stdio, following the MCP (Model Context Protocol) specification.

## Contributing

Contributions are welcome\! Please see the main Smart Fork repository for contribution guidelines.

## License

See the main Smart Fork repository for license information.
