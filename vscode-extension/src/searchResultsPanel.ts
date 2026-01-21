import * as vscode from 'vscode';

export class SearchResultsPanel {
    public static currentPanel: SearchResultsPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];

    private constructor(panel: vscode.WebviewPanel, private extensionUri: vscode.Uri) {
        this.panel = panel;
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.panel.webview.html = this.getEmptyHtml();
    }

    public static createOrShow(extensionUri: vscode.Uri): SearchResultsPanel {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (SearchResultsPanel.currentPanel) {
            SearchResultsPanel.currentPanel.panel.reveal(column);
            return SearchResultsPanel.currentPanel;
        }

        const panel = vscode.window.createWebviewPanel(
            'smartForkResults',
            'Smart Fork: Search Results',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [extensionUri]
            }
        );

        SearchResultsPanel.currentPanel = new SearchResultsPanel(panel, extensionUri);
        return SearchResultsPanel.currentPanel;
    }

    public showResults(query: string, results: string): void {
        this.panel.webview.html = this.getResultsHtml(query, results);
    }

    public showLoading(query: string): void {
        this.panel.webview.html = this.getLoadingHtml(query);
    }

    public showError(error: string): void {
        this.panel.webview.html = this.getErrorHtml(error);
    }

    private getEmptyHtml(): string {
        return `<\!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Smart Fork</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
            line-height: 1.6;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--vscode-descriptionForeground);
        }
        .empty-state h2 {
            color: var(--vscode-foreground);
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class='empty-state'>
        <h2>Smart Fork Search</h2>
        <p>Use the 'Smart Fork: Search Sessions' command to find relevant past work</p>
    </div>
</body>
</html>`;
    }

    private getLoadingHtml(query: string): string {
        const escapedQuery = this.escapeHtml(query);
        return `<\!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Searching...</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
        }
        .loading {
            text-align: center;
            padding: 40px;
        }
        .spinner {
            border: 3px solid var(--vscode-editorWidget-background);
            border-top: 3px solid var(--vscode-progressBar-background);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class='loading'>
        <h2>Searching for: '${escapedQuery}'</h2>
        <div class='spinner'></div>
        <p>Searching through your Claude Code sessions...</p>
    </div>
</body>
</html>`;
    }

    private getResultsHtml(query: string, results: string): string {
        const escapedQuery = this.escapeHtml(query);
        const escapedResults = this.escapeHtml(results);
        
        return `<\!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Search Results</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
            line-height: 1.6;
        }
        .header {
            border-bottom: 1px solid var(--vscode-panel-border);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .header h2 {
            margin: 0 0 5px 0;
            color: var(--vscode-foreground);
        }
        .query {
            color: var(--vscode-descriptionForeground);
            font-style: italic;
        }
        .results {
            white-space: pre-wrap;
            font-family: var(--vscode-editor-font-family);
            font-size: var(--vscode-editor-font-size);
        }
        .no-results {
            padding: 40px 20px;
            text-align: center;
            color: var(--vscode-descriptionForeground);
        }
    </style>
</head>
<body>
    <div class='header'>
        <h2>Search Results</h2>
        <div class='query'>Query: '${escapedQuery}'</div>
    </div>
    <div class='results'>${escapedResults}</div>
</body>
</html>`;
    }

    private getErrorHtml(error: string): string {
        const escapedError = this.escapeHtml(error);
        return `<\!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Error</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
        }
        .error {
            background-color: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-inputValidation-errorBorder);
            padding: 20px;
            border-radius: 4px;
        }
        .error h2 {
            margin-top: 0;
            color: var(--vscode-errorForeground);
        }
        .error-message {
            white-space: pre-wrap;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class='error'>
        <h2>Error</h2>
        <div class='error-message'>${escapedError}</div>
    </div>
</body>
</html>`;
    }

    private escapeHtml(text: string): string {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/'/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    public dispose(): void {
        SearchResultsPanel.currentPanel = undefined;
        this.panel.dispose();
        while (this.disposables.length) {
            const disposable = this.disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }
}
