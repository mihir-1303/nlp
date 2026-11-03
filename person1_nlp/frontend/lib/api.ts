export type HealthResponse = {
  status: "ok" | "degraded";
  models: {
    nlp_model_loaded: boolean;
    delivery_model_loaded: boolean;
  };
};

export type IncidentClassificationResponse = {
  predicted_category: string;
  model_confidence: number;
  confidence_message: string;
};

export type DeliveryPredictionRequest = {
  "Days for shipment (scheduled)": number;
  "Order Item Quantity": number;
  "Order Item Product Price": number;
  "Order Item Discount": number;
  "Order Item Discount Rate": number;
  Latitude: number;
  Longitude: number;
  Type: string;
  "Shipping Mode": string;
  Market: string;
  "Order Region": string;
  "Category Name": string;
  "Department Name": string;
  "Customer Segment": string;
  threshold?: number;
};

export type DeliveryPredictionResponse = {
  estimated_late_delivery_probability: number;
  prediction: 0 | 1;
  warning: boolean;
  threshold_used: number;
  target: "Late_delivery_risk";
};

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error("Could not connect to the backend. Make sure FastAPI is running.");
  }

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Backend request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  classifyIncident: (description: string) =>
    request<IncidentClassificationResponse>("/api/incidents/classify", {
      method: "POST",
      body: JSON.stringify({ description }),
    }),
  predictDelivery: (payload: DeliveryPredictionRequest) =>
    request<DeliveryPredictionResponse>("/api/delivery/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
