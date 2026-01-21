import { spawn, ChildProcessWithoutNullStreams } from 'child_process';

export interface MCPToolCall {
    name: string;
    arguments: Record<string, any>;
}

export interface MCPResponse {
    content: Array<{
        type: string;
        text: string;
    }>;
}

export class MCPClient {
    private process: ChildProcessWithoutNullStreams | null = null;
    private requestId = 0;
    private pendingRequests: Map<number, {
        resolve: (value: any) => void;
        reject: (reason: any) => void;
    }> = new Map();
    private initialized = false;
    private buffer = '';

    constructor(
        private pythonPath: string = 'python',
        private serverModule: string = '-m smart_fork.server'
    ) {}

    async start(): Promise<void> {
        if (this.process) {
            return;
        }

        return new Promise((resolve, reject) => {
            const args = this.serverModule.split(' ');
            this.process = spawn(this.pythonPath, args, {
                stdio: ['pipe', 'pipe', 'pipe']
            });

            if (\!this.process.stdout || \!this.process.stdin) {
                reject(new Error('Failed to create process streams'));
                return;
            }

            this.process.stdout.on('data', (data) => {
                this.buffer += data.toString();
                const lines = this.buffer.split('
');
                this.buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.trim()) {
                        this.handleResponse(JSON.parse(line));
                    }
                }
            });

            this.process.stderr.on('data', (data) => {
                console.error('MCP Server:', data.toString());
            });

            this.process.on('error', (error) => {
                reject(error);
            });

            this.process.on('exit', (code) => {
                console.log(`MCP Server exited with code ${code}`);
                this.process = null;
                this.initialized = false;
            });

            this.initialize().then(() => {
                this.initialized = true;
                resolve();
            }).catch(reject);
        });
    }

    private async initialize(): Promise<void> {
        const response = await this.sendRequest('initialize', {
            protocolVersion: '2024-11-05',
            capabilities: {},
            clientInfo: {
                name: 'vscode-smart-fork',
                version: '0.1.0'
            }
        });

        await this.sendNotification('notifications/initialized', {});
        return response;
    }

    async listTools(): Promise<any[]> {
        if (\!this.initialized) {
            await this.start();
        }

        const response = await this.sendRequest('tools/list', {});
        return response.tools || [];
    }

    async callTool(toolName: string, args: Record<string, any>): Promise<MCPResponse> {
        if (\!this.initialized) {
            await this.start();
        }

        const response = await this.sendRequest('tools/call', {
            name: toolName,
            arguments: args
        });

        return response;
    }

    private sendRequest(method: string, params: any): Promise<any> {
        return new Promise((resolve, reject) => {
            const id = ++this.requestId;
            const request = {
                jsonrpc: '2.0',
                id,
                method,
                params
            };

            this.pendingRequests.set(id, { resolve, reject });

            if (this.process?.stdin) {
                this.process.stdin.write(JSON.stringify(request) + '
');
            } else {
                reject(new Error('Process not started'));
            }

            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error('Request timeout'));
                }
            }, 30000);
        });
    }

    private sendNotification(method: string, params: any): Promise<void> {
        return new Promise((resolve, reject) => {
            const notification = {
                jsonrpc: '2.0',
                method,
                params
            };

            if (this.process?.stdin) {
                this.process.stdin.write(JSON.stringify(notification) + '
');
                resolve();
            } else {
                reject(new Error('Process not started'));
            }
        });
    }

    private handleResponse(response: any): void {
        if (response.id \!== undefined && this.pendingRequests.has(response.id)) {
            const { resolve, reject } = this.pendingRequests.get(response.id)\!;
            this.pendingRequests.delete(response.id);

            if (response.error) {
                reject(new Error(response.error.message));
            } else {
                resolve(response.result);
            }
        }
    }

    async stop(): Promise<void> {
        if (this.process) {
            this.process.kill();
            this.process = null;
            this.initialized = false;
        }
    }
}
