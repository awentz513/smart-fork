import * as vscode from 'vscode';
import { MCPClient } from './mcpClient';
import { SearchResultsPanel } from './searchResultsPanel';

let mcpClient: MCPClient | undefined;
let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('Smart Fork');
    outputChannel.appendLine('Smart Fork extension activated');

    // Register search command
    const searchCommand = vscode.commands.registerCommand('smart-fork.search', async () => {
        try {
            const query = await vscode.window.showInputBox({
                prompt: 'Enter search query to find relevant past work',
                placeHolder: 'e.g., authentication API, React component, database migration',
                ignoreFocusOut: true
            });

            if (\!query) {
                return;
            }

            // Show loading state
            const panel = SearchResultsPanel.createOrShow(context.extensionUri);
            panel.showLoading(query);

            // Get or create MCP client
            if (\!mcpClient) {
                mcpClient = await createMCPClient();
            }

            // Call fork-detect tool
            const response = await mcpClient.callTool('fork-detect', { query });
            
            // Display results
            if (response.content && response.content.length > 0) {
                const resultsText = response.content[0].text;
                panel.showResults(query, resultsText);
            } else {
                panel.showResults(query, 'No results found');
            }

        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            vscode.window.showErrorMessage(`Smart Fork search failed: ${message}`);
            outputChannel.appendLine(`Search error: ${message}`);
            
            const panel = SearchResultsPanel.currentPanel;
            if (panel) {
                panel.showError(message);
            }
        }
    });

    // Register fork command
    const forkCommand = vscode.commands.registerCommand('smart-fork.fork', async () => {
        try {
            const sessionId = await vscode.window.showInputBox({
                prompt: 'Enter session ID to fork from',
                placeHolder: 'e.g., 2024-01-15T10-30-45-abc123',
                ignoreFocusOut: true
            });

            if (\!sessionId) {
                return;
            }

            // Get or create MCP client
            if (\!mcpClient) {
                mcpClient = await createMCPClient();
            }

            // Get session preview first
            const previewResponse = await mcpClient.callTool('get-session-preview', { 
                session_id: sessionId,
                length: 500
            });

            if (previewResponse.content && previewResponse.content.length > 0) {
                const previewText = previewResponse.content[0].text;
                
                // Show preview and confirm
                const selection = await vscode.window.showInformationMessage(
                    `Fork from session: ${sessionId}`,
                    { modal: true, detail: previewText },
                    'Fork',
                    'Cancel'
                );

                if (selection === 'Fork') {
                    // In a real implementation, this would trigger the fork
                    // For now, just show a message with the fork command
                    const forkCommand = `claude fork ${sessionId}`;
                    
                    const copySelection = await vscode.window.showInformationMessage(
                        `To fork this session, run the following command in Claude Code:`,
                        { modal: true, detail: forkCommand },
                        'Copy Command'
                    );

                    if (copySelection === 'Copy Command') {
                        await vscode.env.clipboard.writeText(forkCommand);
                        vscode.window.showInformationMessage('Fork command copied to clipboard');
                    }
                }
            }

        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            vscode.window.showErrorMessage(`Smart Fork fork failed: ${message}`);
            outputChannel.appendLine(`Fork error: ${message}`);
        }
    });

    // Register history command
    const historyCommand = vscode.commands.registerCommand('smart-fork.history', async () => {
        try {
            // Get or create MCP client
            if (\!mcpClient) {
                mcpClient = await createMCPClient();
            }

            // Call get-fork-history tool
            const response = await mcpClient.callTool('get-fork-history', {});
            
            if (response.content && response.content.length > 0) {
                const historyText = response.content[0].text;
                
                // Create a new document to show the history
                const doc = await vscode.workspace.openTextDocument({
                    content: historyText,
                    language: 'markdown'
                });
                
                await vscode.window.showTextDocument(doc, {
                    preview: true,
                    viewColumn: vscode.ViewColumn.Beside
                });
            } else {
                vscode.window.showInformationMessage('No fork history found');
            }

        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            vscode.window.showErrorMessage(`Smart Fork history failed: ${message}`);
            outputChannel.appendLine(`History error: ${message}`);
        }
    });

    // Auto-start MCP server if configured
    const config = vscode.workspace.getConfiguration('smart-fork');
    if (config.get('autoStart', true)) {
        createMCPClient().then(client => {
            mcpClient = client;
            outputChannel.appendLine('MCP client started automatically');
        }).catch(error => {
            outputChannel.appendLine(`Failed to auto-start MCP client: ${error.message}`);
        });
    }

    context.subscriptions.push(searchCommand, forkCommand, historyCommand, outputChannel);
}

async function createMCPClient(): Promise<MCPClient> {
    const config = vscode.workspace.getConfiguration('smart-fork');
    const pythonPath = config.get('pythonPath', 'python');
    const serverPath = config.get('serverPath', '-m smart_fork.server');

    outputChannel.appendLine(`Creating MCP client with python: ${pythonPath}, server: ${serverPath}`);

    const client = new MCPClient(pythonPath, serverPath);
    await client.start();

    outputChannel.appendLine('MCP client started successfully');
    return client;
}

export function deactivate() {
    if (mcpClient) {
        mcpClient.stop();
    }
    outputChannel.appendLine('Smart Fork extension deactivated');
}
