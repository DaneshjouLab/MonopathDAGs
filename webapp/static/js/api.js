// api.js - handles backend communication

export async function fetchGraphData() {
    const response = await fetch("/graph-data");
    if (!response.ok) {
        throw new Error("Failed to fetch graph data");
    }
    return await response.json();
}
