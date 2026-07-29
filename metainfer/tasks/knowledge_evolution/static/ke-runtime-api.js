/** Fetch helpers for knowledge-evolution task endpoints.
 *
 * Each endpoint hits /api/knowledge-evolution/{taskId}/...
 * Auto-discovered by importmap: ke-runtime-api.js -> app/ke-runtime-api
 */

const TASK_BASE = (taskId) => `/api/knowledge-evolution/${taskId}`;

export async function getIterations(taskId) {
    const resp = await fetch(`${TASK_BASE(taskId)}/iterations`);
    if (!resp.ok) throw new Error(`GET iterations failed: ${resp.status}`);
    return resp.json();
}

export async function getStateGraph(taskId) {
    const resp = await fetch(`${TASK_BASE(taskId)}/state-graph`, { cache: "no-store" });
    if (!resp.ok) throw new Error(`GET state-graph failed: ${resp.status}`);
    return resp.json();
}

export async function getKnowledgeGained(taskId) {
    const resp = await fetch(`${TASK_BASE(taskId)}/knowledge-gained`);
    if (!resp.ok) throw new Error(`GET knowledge-gained failed: ${resp.status}`);
    return resp.json();
}

export async function getModelCode(taskId) {
    const resp = await fetch(`${TASK_BASE(taskId)}/model-code`);
    if (!resp.ok) throw new Error(`GET model-code failed: ${resp.status}`);
    return resp.json();
}

export async function getModelCodeFile(taskId, iteration, file) {
    const resp = await fetch(`${TASK_BASE(taskId)}/model-code-file?iteration=${iteration}&file=${encodeURIComponent(file)}`);
    if (!resp.ok) throw new Error(`GET model-code-file failed: ${resp.status}`);
    return resp.json();
}

export async function getIterationDetail(taskId, iteration) {
    const resp = await fetch(`${TASK_BASE(taskId)}/iterations/${iteration}/detail`);
    if (!resp.ok) throw new Error(`GET iteration-detail failed: ${resp.status}`);
    return resp.json();
}

export async function getOracleReport(taskId, iteration) {
    const resp = await fetch(`${TASK_BASE(taskId)}/iterations/${iteration}/oracle-report`);
    if (!resp.ok) throw new Error(`GET oracle-report failed: ${resp.status}`);
    return resp.json();
}

export async function getKnowledgeDiff(taskId, iteration, file) {
    const resp = await fetch(`${TASK_BASE(taskId)}/knowledge-diff?iteration=${iteration}&file=${encodeURIComponent(file)}`);
    if (!resp.ok) throw new Error(`GET knowledge-diff failed: ${resp.status}`);
    return resp.json();
}

export async function getRetrospective(taskId, iteration) {
    const resp = await fetch(`${TASK_BASE(taskId)}/iterations/${iteration}/retrospective`);
    if (!resp.ok) throw new Error(`GET retrospective failed: ${resp.status}`);
    return resp.json();
}

export async function getLog(taskId) {
    const resp = await fetch(`${TASK_BASE(taskId)}/log`);
    if (!resp.ok) throw new Error(`GET log failed: ${resp.status}`);
    return resp.json();
}

export async function getCharts(taskId) {
    const resp = await fetch(`${TASK_BASE(taskId)}/charts`);
    if (!resp.ok) throw new Error(`GET charts failed: ${resp.status}`);
    return resp.json();
}

export async function getAgentStatus(taskId) {
    const resp = await fetch(`${TASK_BASE(taskId)}/agent-status`);
    if (!resp.ok) throw new Error(`GET agent-status failed: ${resp.status}`);
    const data = await resp.json();
    return data.status || "";
}
