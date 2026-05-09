// ── Configura aquí la IP local de tu servidor FastAPI ──────────────────────
// Ejecuta `ipconfig` (Windows) o `ifconfig` (Mac/Linux) y pon la IPv4 local.
// Ejemplo: 'http://192.168.1.45:8000'
export const API_BASE_URL = 'http://195.57.190.10:8000';

export interface PredictionResult {
  predicted_class: string;
  raw_predicted_class: string;
  confidence: number;
  is_uncertain: boolean;
  threshold: number;
  gradcam_overlay_base64: string | null;
}

// Respuestas mock para cuando el servidor no está disponible
const MOCK_RESPONSES: PredictionResult[] = [
  {
    predicted_class: 'tipo1',
    raw_predicted_class: 'tipo1',
    confidence: 0.943,
    is_uncertain: false,
    threshold: 0.7,
    gradcam_overlay_base64: null,
  },
  {
    predicted_class: 'tipo2',
    raw_predicted_class: 'tipo2',
    confidence: 0.871,
    is_uncertain: false,
    threshold: 0.7,
    gradcam_overlay_base64: null,
  },
  {
    predicted_class: 'tipo3',
    raw_predicted_class: 'tipo3',
    confidence: 0.912,
    is_uncertain: false,
    threshold: 0.7,
    gradcam_overlay_base64: null,
  },
];

function getMockResponse(): PredictionResult {
  return MOCK_RESPONSES[Math.floor(Math.random() * MOCK_RESPONSES.length)];
}

export async function predictLeaf(imageUri: string): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    name: 'leaf.jpg',
    type: 'image/jpeg',
  } as unknown as Blob);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12_000);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return (await response.json()) as PredictionResult;
  } catch (error) {
    clearTimeout(timeout);
    // En modo demo devuelve resultado mock si el servidor no está disponible
    console.warn('API no disponible, usando respuesta de demostración:', error);
    return getMockResponse();
  }
}
