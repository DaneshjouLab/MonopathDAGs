import { Network } from "https://esm.sh/vis-network/peer";
import { DataSet } from "https://esm.sh/vis-data/peer";
import { fetchGraphData } from "./api.js";

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const graph = await fetchGraphData();
        initGraph(graph);
    } catch (error) {
        console.error("Error initializing graph:", error);
    }
});

function initGraph(graph) {
    console.log(graph.nodes)
    const nodes = new DataSet(graph.nodes);
    const edges = new DataSet(graph.edges);

    const container = document.getElementById('mynetwork');
    const data = { nodes, edges };
    const options = {
        interaction: {
            multiselect: false,
            hover: true  // <-- REQUIRED for hoverNode and hoverEdge
        }
    };

    const network = new Network(container, data, options);

    network.on("select", (params) => {
        updateSidebar(params, nodes, edges);
    });
    network.on("hoverNode", function (params) {
        console.log("Hovering over node:", params.node);
    });
}

function updateSidebar(params, nodes, edges) {
    const infoDiv = document.getElementById('info');

    let selectedData = null;

    if (params.nodes.length > 0) {
        const node = nodes.get(params.nodes[0]);
        selectedData = node;
    } else if (params.edges.length > 0) {
        const edge = edges.get(params.edges[0]);
        selectedData = edge;
    }

    if (selectedData) {
        const { id, label, from, to, customData } = selectedData;

        let content = `<b>${label || 'Edge Selected'}</b><br>`;
        if (id !== undefined) content += `ID: ${id}<br>`;
        if (from !== undefined && to !== undefined) content += `From: ${from} → To: ${to}<br>`;

        content += `<b>Data:</b><br>${formatCustomData(customData)}`;
        infoDiv.innerHTML = content;

        sendSelectionToServer({
            id,
            label,
            from,
            to,
            customData
        });
    
        
    } else {
        infoDiv.innerHTML = 'Click a node or edge to view details here.';
    }
}


function formatCustomData(data) {
    if (!data || typeof data !== 'object') {
        return `<i>No additional data</i>`;
    }

    const entries = Object.entries(data).map(([key, value]) => {
        if (typeof value === 'object') {
            return `<div><b>${key}:</b><pre>${JSON.stringify(value, null, 2)}</pre></div>`;
        } else {
            return `<div><b>${key}:</b> ${value}</div>`;
        }
    });

    return entries.join('');
}
async function sendSelectionToServer(data) {
    try {
        const response = await fetch('/api/selection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            console.error('Failed to send selection to server:', response.statusText);
        }
    } catch (error) {
        console.error('Error sending selection to server:', error);
    }
}
