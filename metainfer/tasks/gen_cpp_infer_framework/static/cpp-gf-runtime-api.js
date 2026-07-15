// Fetch helpers for gen_cpp_infer_framework's task-specific endpoints.
//
// All routes live under /api/gen-cpp-infer-framework/<id>/* (mounted by
// the shell from this task's server router). The shell api.js
// intentionally does NOT know about these — every task package owns
// its own runtime fetcher.

const TASK_BASE = (taskId) =>
  `/api/gen-cpp-infer-framework/${encodeURIComponent(taskId)}`;

export async function getIterations(taskId) {
  const r = await fetch(`${TASK_BASE(taskId)}/iterations`);
  if (!r.ok) throw new Error(`iterations: ${r.status}`);
  return r.json();
}

export async function getIteration(taskId, n) {
  const r = await fetch(`${TASK_BASE(taskId)}/iterations/${n}`);
  if (!r.ok) throw new Error(`iteration ${n}: ${r.status}`);
  return r.json();
}

export async function getRetrospective(taskId, n) {
  const r = await fetch(`${TASK_BASE(taskId)}/iterations/${n}/retrospective`);
  if (!r.ok) throw new Error(`retro ${n}: ${r.status}`);
  return r.json();
}

export async function getCharts(taskId) {
  const r = await fetch(`${TASK_BASE(taskId)}/charts`);
  if (!r.ok) throw new Error(`charts: ${r.status}`);
  return r.json();
}

export async function getStateGraph(taskId) {
  const r = await fetch(`${TASK_BASE(taskId)}/state-graph`,
                        { cache: "no-store" });
  if (!r.ok) throw new Error(`state-graph: ${r.status}`);
  return r.json();
}
