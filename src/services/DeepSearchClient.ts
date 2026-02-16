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
    if (baseUrl) {
      this.baseUrl = baseUrl;
    } else if (import.meta.env.VITE_DEEPSEARCH_API_URL) {
      this.baseUrl = import.meta.env.VITE_DEEPSEARCH_API_URL as string;
    } else {
      // Default to current origin if not specified (for production)
      // If we are in development (localhost:3000/5173), we might want to default to localhost:8000
      // but usually VITE_DEEPSEARCH_API_URL should be set in .env for dev.
      // For Docker/Production where frontend is served by backend, empty string implies relative path.
      this.baseUrl = "";
    }

    if (this.baseUrl.startsWith('http')) {
      this.wsUrl = this.baseUrl.replace(/^http/, 'ws') + '/ws';
    } else if (this.baseUrl === "") {
        // Relative path, construct absolute WS URL based on current location
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.wsUrl = `${protocol}//${window.location.host}/ws`;
    } else {
        // Fallback or assuming baseUrl is just a path like "/api"
         const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
         this.wsUrl = `${protocol}//${window.location.host}${this.baseUrl}/ws`;
    }
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
