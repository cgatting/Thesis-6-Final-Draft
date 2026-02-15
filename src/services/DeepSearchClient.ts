export interface DeepSearchRefineResponse {
  processedText: string;
  bibliographyText: string;
  bibtex: string;
}

export type ProgressCallback = (percent: number, message: string) => void;

export class DeepSearchClient {
  private baseUrl: string;
  private wsUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || (import.meta.env.VITE_DEEPSEARCH_API_URL as string) || "http://localhost:8000";
    this.wsUrl = this.baseUrl.replace(/^http/, 'ws') + '/ws';
  }

  public async refineDocument(manuscriptText: string, onProgress?: ProgressCallback): Promise<DeepSearchRefineResponse> {
    let socket: WebSocket | null = null;

    if (onProgress) {
      try {
        socket = new WebSocket(this.wsUrl);
        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'progress') {
              onProgress(data.progress * 100, data.message);
            } else if (data.type === 'error') {
              console.error("DeepSearch Error:", data.message);
            }
          } catch (e) {
            console.warn("Failed to parse WS message:", event.data);
          }
        };
        
        // Wait for connection to open
        await new Promise<void>((resolve) => {
          if (!socket) return resolve();
          if (socket.readyState === WebSocket.OPEN) return resolve();
          socket.onopen = () => resolve();
          // Timeout after 2s if WS fails, proceed anyway
          setTimeout(resolve, 2000); 
        });
      } catch (e) {
        console.warn("WebSocket connection failed, proceeding without progress updates", e);
      }
    }

    try {
      const response = await fetch(`${this.baseUrl}/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ manuscriptText })
      });

      if (!response.ok) {
        throw new Error(`DeepSearch API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } finally {
      if (socket) {
        socket.close();
      }
    }
  }
}
