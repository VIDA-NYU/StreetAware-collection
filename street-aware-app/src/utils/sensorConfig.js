const BASE_URL = 'http://localhost:8080';

export async function fetchSensors() {
  const response = await fetch(`${BASE_URL}/sensors`);
  if (!response.ok) {
    throw new Error('Failed to fetch sensor configuration');
  }
  return response.json();
}
