const apiBase = import.meta.env.VITE_API_URL || "/api";

export async function getHealth() {
  const response = await fetch(`${apiBase}/health`);
  if (!response.ok) throw new Error(`Health check failed (${response.status})`);
  return response.json();
}

export async function predictAll(payload) {
  const response = await fetch(`${apiBase}/predict/all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `Prediction failed (${response.status})`);
  }
  return body;
}
