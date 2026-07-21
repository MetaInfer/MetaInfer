// Runtime API client for evolve-kernel task type.
// All fetches go through /api/evolve-kernel/{task_id}/...

const BASE = (taskId) => `/api/evolve-kernel/${taskId}`;

export async function getIterations(taskId) {
  const res = await fetch(`${BASE(taskId)}/iterations`);
  if (!res.ok) throw new Error(`iterations: ${res.status}`);
  return res.json();
}

export async function getIterationDetail(taskId, n) {
  const res = await fetch(`${BASE(taskId)}/iterations/${n}`);
  if (!res.ok) throw new Error(`iteration ${n}: ${res.status}`);
  return res.json();
}

export async function getCharts(taskId) {
  const res = await fetch(`${BASE(taskId)}/charts`);
  if (!res.ok) throw new Error(`charts: ${res.status}`);
  return res.json();
}

export async function getStateGraph(taskId) {
  const res = await fetch(`${BASE(taskId)}/state-graph`);
  if (!res.ok) throw new Error(`state-graph: ${res.status}`);
  return res.json();
}

export async function getKernelLibrary(taskId) {
  const res = await fetch(`${BASE(taskId)}/kernel-library`);
  if (!res.ok) throw new Error(`kernel-library: ${res.status}`);
  return res.json();
}

export async function getCorrectnessHarness(taskId) {
  const res = await fetch(`${BASE(taskId)}/harnesses/correctness`);
  if (!res.ok) throw new Error(`correctness harness: ${res.status}`);
  return res.json();
}

export async function getPerfHarness(taskId) {
  const res = await fetch(`${BASE(taskId)}/harnesses/perf`);
  if (!res.ok) throw new Error(`perf harness: ${res.status}`);
  return res.json();
}

export async function getReferenceKernel(taskId) {
  const res = await fetch(`${BASE(taskId)}/reference-kernel`);
  if (!res.ok) throw new Error(`reference kernel: ${res.status}`);
  return res.json();
}

export async function getRetrospective(taskId, n) {
  const res = await fetch(`${BASE(taskId)}/iterations/${n}/retrospective`);
  if (!res.ok) throw new Error(`retrospective ${n}: ${res.status}`);
  return res.json();
}
