// Operator-specific guided shape input for dcu-kernel-auto-opt.
// Self-registers via globalThis.__metainferOverrides — no imports from
// shared modules, no circular deps, no dynamic import needed.
//
// The form-renderer polls __metainferOverrides and picks us up once this
// file is loaded as a side-effect of the importmap entry.

import { html } from "htm/preact";
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";

// ---- helpers ---------------------------------------------------------------

function parseCsvInts(raw) {
  if (!raw || !raw.trim()) return [];
  return raw
    .split(",")
    .map(function (s) { var n = parseInt(s.trim(), 10); return (Number.isFinite(n) && n > 0) ? n : null; })
    .filter(function (n) { return n !== null; });
}

// Frontend mirror of the W8A8 GEMM workload catalog per model. The backend
// validates every submitted shape against the frozen operator contract, so a
// stale UI list fails closed instead of starting an invalid optimization.
var W8A8_M_VALUES = [2, 16, 3072];
// Large-prefill boundary added on 2026-08-06 for TP4; extended to the
// Hy3 / MiniMax M3 / GLM5.2 TP8 catalogs on 2026-08-27 via per-topology
// mValues. DeepSeek TP1/TP8 keep the three original M values. The backend
// validates shapes against the contract.
var W8A8_TP4_M_VALUES = [2, 16, 3072, 4096];

// Model label → W8A8 GEMM workload. (M, K) @ (K, N) per weight name and TP
// size, mirroring the model cards in the operator docs. DeepSeek keeps the
// historical ids (tp4_wqkv_a_m4096 …); other models get a model prefix so ids
// never collide across models.
var MODEL_WORKLOADS = {
  "DeepSeek V4 Flash": {
    idPrefix: "",
    topologies: [
      {
        tp_size: 1,
        operators: [
          { operator: "wqkv_a", K: 4096, N: 1536 },
          { operator: "wq_b", K: 1024, N: 32768 },
          { operator: "indexer.wq_b", K: 1024, N: 8192 },
          { operator: "wo_b", K: 8192, N: 4096 },
          { operator: "shared_gate_up_proj", K: 4096, N: 4096 },
          { operator: "shared_down_proj", K: 2048, N: 4096 },
        ],
      },
      {
        tp_size: 4,
        operators: [
          { operator: "wqkv_a", K: 4096, N: 1536 },
          { operator: "wq_b", K: 1024, N: 8192 },
          { operator: "indexer.wq_b", K: 1024, N: 8192 },
          { operator: "wo_b", K: 2048, N: 4096 },
          { operator: "shared_gate_up_proj", K: 4096, N: 1024 },
          { operator: "shared_down_proj", K: 512, N: 4096 },
        ],
      },
      {
        tp_size: 8,
        operators: [
          { operator: "wqkv_a", K: 4096, N: 1536 },
          { operator: "wq_b", K: 1024, N: 4096 },
          { operator: "indexer.wq_b", K: 1024, N: 8192 },
          { operator: "wo_b", K: 1024, N: 4096 },
          { operator: "shared_gate_up_proj", K: 4096, N: 512 },
          { operator: "shared_down_proj", K: 256, N: 4096 },
        ],
      },
    ],
  },
  "Hy3 (Hunyuan 3)": {
    idPrefix: "hy3_",
    topologies: [
      {
        tp_size: 1,
        operators: [
          { operator: "qkv_proj", K: 4096, N: 10240 },
          { operator: "o_proj", K: 8192, N: 4096 },
          { operator: "shared_gate_up_proj", K: 4096, N: 3072 },
          { operator: "shared_down_proj", K: 1536, N: 4096 },
        ],
      },
      {
        tp_size: 4,
        operators: [
          { operator: "qkv_proj", K: 4096, N: 2560 },
          { operator: "o_proj", K: 2048, N: 4096 },
          { operator: "shared_gate_up_proj", K: 4096, N: 768 },
          { operator: "shared_down_proj", K: 384, N: 4096 },
        ],
      },
      {
        tp_size: 8,
        mValues: [2, 16, 3072, 4096],
        operators: [
          { operator: "qkv_proj", K: 4096, N: 1280 },
          { operator: "o_proj", K: 1024, N: 4096 },
          { operator: "shared_gate_up_proj", K: 4096, N: 384 },
          { operator: "shared_down_proj", K: 192, N: 4096 },
        ],
      },
    ],
  },
  "MiniMax M3": {
    idPrefix: "minimax_",
    topologies: [
      {
        tp_size: 1,
        operators: [
          { operator: "qkv_proj", K: 6144, N: 9216 },
          { operator: "qkv_proj_and_indexer_qk", K: 6144, N: 9856 },
          { operator: "o_proj", K: 8192, N: 6144 },
          { operator: "shared_gate_up_proj", K: 6144, N: 6144 },
          { operator: "shared_down_proj", K: 3072, N: 6144 },
        ],
      },
      {
        tp_size: 4,
        operators: [
          { operator: "qkv_proj", K: 6144, N: 2304 },
          { operator: "qkv_proj_and_indexer_qk", K: 6144, N: 2560 },
          { operator: "o_proj", K: 2048, N: 6144 },
          { operator: "shared_gate_up_proj", K: 6144, N: 1536 },
          { operator: "shared_down_proj", K: 768, N: 6144 },
        ],
      },
      {
        tp_size: 8,
        mValues: [2, 16, 3072, 4096],
        operators: [
          { operator: "qkv_proj", K: 6144, N: 1280 },
          { operator: "qkv_proj_and_indexer_qk", K: 6144, N: 1536 },
          { operator: "o_proj", K: 1024, N: 6144 },
          { operator: "shared_gate_up_proj", K: 6144, N: 768 },
          { operator: "shared_down_proj", K: 384, N: 6144 },
        ],
      },
    ],
  },
  "GLM5.2": {
    idPrefix: "glm_",
    topologies: [
      {
        tp_size: 1,
        operators: [
          { operator: "fused_qkv_a_proj", K: 6144, N: 2624 },
          { operator: "q_b_proj", K: 2048, N: 16384 },
          { operator: "kv_b_proj", K: 512, N: 28672 },
          { operator: "o_proj", K: 16384, N: 6144 },
          { operator: "shared_gate_up_proj", K: 6144, N: 4096 },
          { operator: "shared_down_proj", K: 2048, N: 6144 },
        ],
      },
      {
        tp_size: 4,
        operators: [
          { operator: "fused_qkv_a_proj", K: 6144, N: 2624 },
          { operator: "q_b_proj", K: 2048, N: 4096 },
          { operator: "kv_b_proj", K: 512, N: 7168 },
          { operator: "o_proj", K: 4096, N: 6144 },
          { operator: "shared_gate_up_proj", K: 6144, N: 1024 },
          { operator: "shared_down_proj", K: 512, N: 6144 },
        ],
      },
      {
        tp_size: 8,
        mValues: [2, 16, 3072, 4096],
        operators: [
          { operator: "fused_qkv_a_proj", K: 6144, N: 2624 },
          { operator: "q_b_proj", K: 2048, N: 2048 },
          { operator: "kv_b_proj", K: 512, N: 3584 },
          { operator: "o_proj", K: 2048, N: 6144 },
          { operator: "shared_gate_up_proj", K: 6144, N: 512 },
          { operator: "shared_down_proj", K: 256, N: 6144 },
        ],
      },
    ],
  },
};

function modelCatalog(modelLabel) {
  var model = MODEL_WORKLOADS[modelLabel] || MODEL_WORKLOADS["DeepSeek V4 Flash"];
  var shapes = [];
  (model.topologies || []).forEach(function (topology) {
    // Per-topology M values override the defaults; TP4 and the Hy3/MiniMax/
    // GLM TP8 catalogs cover the M=4096 large-prefill boundary.
    var mValues = topology.mValues || (topology.tp_size === 4
      ? W8A8_TP4_M_VALUES : W8A8_M_VALUES);
    topology.operators.forEach(function (item) {
      mValues.forEach(function (M) {
        shapes.push({
          id: model.idPrefix + "tp" + topology.tp_size + "_" +
            item.operator.replace(/\./g, "_") + "_m" + M,
          tp_size: topology.tp_size,
          operator: item.operator,
          M: M,
          N: item.N,
          K: item.K,
        });
      });
    });
  });
  return shapes;
}

// Backward-compatible DeepSeek-only default catalog.
var DEFAULT_W8A8_SHAPES = modelCatalog("DeepSeek V4 Flash");

function currentShapeId(catalog, rawId, modelLabel) {
  if (catalog.some(function (shape) { return shape.id === rawId; })) {
    return rawId;
  }
  if (modelLabel === "DeepSeek V4 Flash") {
    var legacy = /^m(\d+)_(wqkv_a|wq_b|wo_b|shared_gate_up|shared_down)$/.exec(rawId);
    if (!legacy) return null;
    var operatorAliases = {
      shared_gate_up: "shared_gate_up_proj",
      shared_down: "shared_down_proj",
    };
    var operator = operatorAliases[legacy[2]] || legacy[2];
    var migrated = "tp4_" + operator + "_m" + legacy[1];
    return catalog.some(function (shape) {
      return shape.id === migrated;
    }) ? migrated : null;
  }
  return null;
}

function selectedShapes(parsedShapes, fallbackToAll, catalog, modelLabel) {
  var ids = parsedShapes
    .map(function (shape) { return currentShapeId(catalog, shape.id, modelLabel); })
    .filter(Boolean);
  if (!ids.length && fallbackToAll) {
    ids = catalog.map(function (shape) { return shape.id; });
  }
  return selectedMap(ids);
}

function parseShapeRecords(raw) {
  if (!raw) return [];
  var shapes = [], re = /\{[^}]+\}/g, match;
  while ((match = re.exec(raw)) !== null) {
    var text = match[0];
    var idMatch = /\bid:\s*["']?([^,\s}"']+)/.exec(text);
    var getInt = function (key) {
      var dimMatch = new RegExp("\\b" + key + ":\\s*(\\d+)").exec(text);
      return dimMatch ? parseInt(dimMatch[1], 10) : null;
    };
    var operatorMatch = /\boperator:\s*["']?([^,\s}"']+)/.exec(text);
    var M = getInt("M"), N = getInt("N"), K = getInt("K");
    var tpSize = getInt("tp_size");
    if (idMatch && M != null && N != null && K != null) {
      shapes.push({
        id: idMatch[1],
        tp_size: tpSize,
        operator: operatorMatch ? operatorMatch[1] : null,
        M: M,
        N: N,
        K: K,
      });
    }
  }
  return shapes;
}

function parseManualOwners(raw) {
  var owners = {};
  if (!raw) return owners;
  var re = /worker_(\d+):\s*\{[^}]*shapes:\s*\[([^\]]*)\]/g, match;
  while ((match = re.exec(raw)) !== null) {
    var gpu = parseInt(match[1], 10);
    match[2].split(",").forEach(function (item) {
      var id = item.trim().replace(/^["']|["']$/g, "");
      if (id) owners[id] = gpu;
    });
  }
  return owners;
}

function operatorGroupedOwners(shapes) {
  var familyGpu = {
    wqkv_a: 0,
    fused_qkv_a_proj: 0,
    qkv_proj: 0,
    wq_b: 1,
    "indexer.wq_b": 1,
    qkv_proj_and_indexer_qk: 1,
    q_b_proj: 1,
    kv_b_proj: 1,
    wo_b: 2,
    o_proj: 2,
    shared_gate_up_proj: 3,
    shared_down_proj: 3,
  };
  var owners = {};
  shapes.forEach(function (shape, index) {
    var family = shape.operator || shape.id;
    owners[shape.id] = familyGpu[family] != null
      ? familyGpu[family]
      : index % 4;
  });
  return owners;
}

function mGroupedOwners(shapes) {
  var mValues = shapes
    .map(function (shape) { return shape.M; })
    .filter(function (value, index, all) {
      return all.indexOf(value) === index;
    })
    .sort(function (a, b) { return a - b; });
  var owners = {};
  shapes.forEach(function (shape) {
    owners[shape.id] = mValues.indexOf(shape.M) % 4;
  });
  return owners;
}

function balancedOwners(shapes) {
  var owners = {};
  shapes.forEach(function (shape, index) {
    owners[shape.id] = index % 4;
  });
  return owners;
}

function serializeShapesHeader(mode, scope, shapes, modelLabel) {
  var lines = [];
  if (modelLabel) lines.push("model: " + modelLabel);
  lines.push(
    "assignment_mode: " + mode,
    "shape_scope: " + scope,
    "shapes:"
  );
  shapes.forEach(function (shape) {
    var logical = shape.tp_size != null && shape.operator
      ? ", tp_size: " + shape.tp_size +
        ", operator: \"" + shape.operator + "\""
      : "";
    lines.push(
      "  - {id: " + shape.id + ", M: " + shape.M +
      ", N: " + shape.N + ", K: " + shape.K + logical + "}"
    );
  });
  return lines;
}

function serializeManualAssignment(shapes, owners, scope, modelLabel) {
  var lines = serializeShapesHeader("manual", scope, shapes, modelLabel);
  lines.push("assignments:");
  for (var gpu = 0; gpu < 4; gpu++) {
    var ids = shapes
      .filter(function (shape) { return owners[shape.id] === gpu; })
      .map(function (shape) { return shape.id; });
    if (ids.length) {
      lines.push(
        "  worker_" + gpu + ": {gpu: " + gpu +
        ", shapes: [" + ids.join(", ") + "]}"
      );
    }
  }
  return lines.join("\n");
}

function selectedMap(ids) {
  var selected = {};
  ids.forEach(function (id) { selected[id] = true; });
  return selected;
}

function TpFilterBar(_a) {
  var tpSizes = _a.tpSizes, active = _a.active, onChange = _a.onChange;
  return html`
    <div class="dkao-tp-filter">
      <span class="dkao-tp-filter-label">TP</span>
      <button type="button" class="dkao-shape-tab ${active == null ? "active" : ""}"
        onClick=${function () { return onChange(null); }}>
        All
      </button>
      ${tpSizes.map(function (tp) { return html`
        <button type="button" class="dkao-shape-tab ${active === tp ? "active" : ""}"
          onClick=${function () { return onChange(tp); }}>
          TP${tp}
        </button>
      `; })}
    </div>
  `;
}

function ShapeCatalog(_a) {
  var shapes = _a.shapes, selected = _a.selected, onToggle = _a.onToggle,
      readOnly = _a.readOnly;
  var tpSizes = shapes
    .map(function (shape) { return shape.tp_size; })
    .filter(function (value, index, all) { return all.indexOf(value) === index; })
    .sort(function (a, b) { return a - b; });
  var _b = useState(null), tpFilter = _b[0], setTpFilter = _b[1];
  var visible = tpFilter == null
    ? shapes
    : shapes.filter(function (shape) { return shape.tp_size === tpFilter; });
  var selectedCount = shapes.filter(function (shape) {
    return selected[shape.id];
  }).length;
  var visibleSelected = visible.filter(function (shape) {
    return selected[shape.id];
  }).length;
  return html`
    <div class="dkao-shape-catalog">
      <div class="dkao-shape-catalog-header">
        <strong>Shapes in this task</strong>
        <span>
          ${selectedCount} / ${shapes.length} selected
          ${tpFilter != null ? ` · ${visibleSelected} shown` : ""}
        </span>
      </div>
      <${TpFilterBar} tpSizes=${tpSizes} active=${tpFilter} onChange=${setTpFilter} />
      <div class="dkao-shape-catalog-items">
        ${visible.map(function (shape) {
          var checked = Boolean(selected[shape.id]);
          return html`
            <label class="dkao-shape-choice ${checked ? "selected" : ""}">
              <input type="checkbox"
                checked=${checked}
                disabled=${readOnly || (checked && selectedCount === 1)}
                onChange=${function () { onToggle(shape.id); }} />
              <span>
                <strong>${shape.id}</strong>
                <small>TP=${shape.tp_size} · ${shape.operator} · M=${shape.M} · N=${shape.N} · K=${shape.K}</small>
              </span>
            </label>
          `;
        })}
        ${visible.length === 0 ? html`<p class="muted">No shapes in this TP size.</p>` : null}
      </div>
    </div>
  `;
}

function ApiDefaultsPreview(_a) {
  var catalog = _a.catalog;
  var tpSizes = catalog
    .map(function (shape) { return shape.tp_size; })
    .filter(function (value, index, all) { return all.indexOf(value) === index; })
    .sort(function (a, b) { return a - b; });
  return html`
    <div class="dkao-manual-assignment">
      <div class="dkao-guided-info">
        <strong>${catalog.length} logical shapes.</strong>
        Workloads are grouped by TP size and kept as separate task identities.
        Each operator is optimized at M=2, M=16 and the large-prefill boundary
        M=3072; TP4 additionally covers M=4096.
      </div>
      <div class="dkao-gpu-grid">
        ${tpSizes.map(function (tp) {
          var topoShapes = catalog.filter(function (shape) {
            return shape.tp_size === tp;
          });
          var operators = [];
          topoShapes.forEach(function (shape) {
            if (!operators.some(function (item) {
              return item.operator === shape.operator;
            })) {
              operators.push({
                operator: shape.operator,
                K: shape.K,
                N: shape.N,
              });
            }
          });
          var mValues = tp === 4 ? W8A8_TP4_M_VALUES : W8A8_M_VALUES;
          return html`
            <section class="dkao-gpu-card">
              <header>
                <strong>TP=${tp}</strong>
                <span>${topoShapes.length} shapes</span>
              </header>
              <div class="dkao-gpu-shapes">
                ${operators.map(function (item) {
                  return html`
                    <div class="dkao-gpu-shape">
                      <div>
                        <strong>${item.operator}</strong>
                        <small>(M,${item.K}) @ (${item.K},${item.N}) · M=${mValues.join(",")}</small>
                      </div>
                    </div>
                  `;
                })}
              </div>
            </section>
          `;
        })}
      </div>
    </div>
  `;
}

function AiShapeSubset(_a) {
  var value = _a.value, onChange = _a.onChange, model = _a.model;
  var catalog = modelCatalog(model);
  var _b = useState(function () {
    var parsed = parseShapeRecords(value || "");
    return selectedShapes(parsed, true, catalog, model);
  }), selected = _b[0], setSelected = _b[1];

  var emit = useCallback(function (next) {
    var shapes = catalog.filter(function (shape) {
      return next[shape.id];
    });
    setSelected(next);
    onChange(serializeShapesHeader("ai", "subset", shapes, model).join("\n"));
  }, [onChange, catalog, model]);

  useEffect(function () { emit(selected); }, []);

  return html`
    <div class="dkao-manual-assignment">
      <div class="dkao-guided-info">
        <strong>Subset task.</strong>
        The control plane assigns only the checked shapes. Unchecked API
        shapes stay on the trusted fallback and are checked again before the
        candidate is published.
      </div>
      <${ShapeCatalog}
        shapes=${catalog}
        selected=${selected}
        onToggle=${function (shapeId) {
          var next = Object.assign({}, selected);
          next[shapeId] = !next[shapeId];
          emit(next);
        }} />
    </div>
  `;
}

function ModelWorkloadAll(_a) {
  var value = _a.value, onChange = _a.onChange, model = _a.model;
  var catalog = modelCatalog(model);
  var selected = selectedMap(catalog.map(function (shape) {
    return shape.id;
  }));
  var emitted = useRef(false);
  useEffect(function () {
    if (emitted.current) return;
    emitted.current = true;
    onChange(serializeShapesHeader("ai", "all", catalog, model).join("\n"));
  }, [catalog, onChange, model]);

  return html`
    <div class="dkao-manual-assignment">
      <div class="dkao-guided-info">
        <strong>Full ${model} workload.</strong>
        Every shape below is part of this task. The control plane balances
        them across the four GPUs automatically; the checkboxes are read-only
        because the scope is “All API shapes”.
      </div>
      <${ShapeCatalog}
        shapes=${catalog}
        selected=${selected}
        readOnly=${true}
        onToggle=${function () {}} />
    </div>
  `;
}

function ManualGpuAssignment(_a) {
  var value = _a.value, onChange = _a.onChange, scopeMode = _a.scopeMode,
      model = _a.model;
  var subset = scopeMode === "Selected shapes only";
  var catalog = modelCatalog(model);
  var _b = useState(function () {
    var parsedShapes = parseShapeRecords(value || "");
    return subset
      ? selectedShapes(parsedShapes, true, catalog, model)
      : selectedMap(catalog.map(function (shape) {
        return shape.id;
      }));
  }), selected = _b[0], setSelected = _b[1];
  var _c = useState(function () {
    var parsed = parseManualOwners(value || "");
    var migrated = operatorGroupedOwners(catalog);
    Object.keys(parsed).forEach(function (shapeId) {
      var current = currentShapeId(catalog, shapeId, model);
      if (current) migrated[current] = parsed[shapeId];
    });
    return migrated;
  }), owners = _c[0], setOwners = _c[1];

  var emitState = useCallback(function (nextOwners, nextSelected) {
    var shapes = catalog.filter(function (shape) {
      return !subset || nextSelected[shape.id];
    });
    setOwners(nextOwners);
    setSelected(nextSelected);
    onChange(serializeManualAssignment(
      shapes, nextOwners, subset ? "subset" : "all", model
    ));
  }, [onChange, subset, catalog, model]);

  useEffect(function () {
    emitState(owners, selected);
  }, [subset]);

  var moveShape = function (shapeId, gpu) {
    var next = Object.assign({}, owners);
    next[shapeId] = gpu;
    emitState(next, selected);
  };
  var shapes = catalog.filter(function (shape) {
    return !subset || selected[shape.id];
  });

  return html`
    <div class="dkao-manual-assignment">
      <div class="dkao-guided-info">
        <strong>Manual assignment is authoritative.</strong>
        The coordinator Agent will not repartition these shapes. Every shape
        appears exactly once; empty GPU cards are allowed.
      </div>
      ${subset && html`
        <${ShapeCatalog}
          shapes=${catalog}
          selected=${selected}
          onToggle=${function (shapeId) {
            var nextSelected = Object.assign({}, selected);
            nextSelected[shapeId] = !nextSelected[shapeId];
            var nextOwners = Object.assign({}, owners);
            if (nextSelected[shapeId] && nextOwners[shapeId] == null) {
              nextOwners[shapeId] = operatorGroupedOwners(
                catalog
              )[shapeId];
            }
            emitState(nextOwners, nextSelected);
          }} />
      `}
      <div class="dkao-assignment-actions">
        <span>Quick layout:</span>
        <button type="button" class="dkao-shape-tab"
          onClick=${function () {
            emitState(operatorGroupedOwners(shapes), selected);
          }}>
          By operator
        </button>
        <button type="button" class="dkao-shape-tab"
          onClick=${function () {
            emitState(mGroupedOwners(shapes), selected);
          }}>
          By M
        </button>
        <button type="button" class="dkao-shape-tab"
          onClick=${function () {
            emitState(balancedOwners(shapes), selected);
          }}>
          Round robin
        </button>
      </div>
      <div class="dkao-gpu-grid">
        ${[0, 1, 2, 3].map(function (gpu) {
          var assigned = shapes.filter(function (shape) {
            return owners[shape.id] === gpu;
          });
          return html`
            <section class="dkao-gpu-card">
              <header>
                <strong>GPU ${gpu}</strong>
                <span>${assigned.length} shapes</span>
              </header>
              <div class="dkao-gpu-shapes">
                ${assigned.length
                  ? assigned.map(function (shape) {
                    return html`
                      <div class="dkao-gpu-shape" key=${shape.id}>
                        <div>
                          <strong>${shape.id}</strong>
                          <small>TP=${shape.tp_size} · ${shape.operator} · M=${shape.M} · N=${shape.N} · K=${shape.K}</small>
                        </div>
                        <select
                          aria-label=${"Move " + shape.id + " to GPU"}
                          value=${String(gpu)}
                          onChange=${function (event) {
                            moveShape(shape.id, parseInt(event.target.value, 10));
                          }}>
                          ${[0, 1, 2, 3].map(function (targetGpu) {
                            return html`
                              <option value=${String(targetGpu)}>
                                GPU ${targetGpu}
                              </option>
                            `;
                          })}
                        </select>
                      </div>
                    `;
                  })
                  : html`<div class="dkao-gpu-empty">No shapes assigned</div>`
                }
              </div>
            </section>
          `;
        })}
      </div>
    </div>
  `;
}

// ---- operator-specific guided forms ----------------------------------------

function GemmGuided(_a) {
  var value = _a.value, onChange = _a.onChange;
  var prev = useMemo(function () { return parseShapeYaml(value || ""); }, [value]);
  var _b = useState(function () { return (prev.mVals || ["2", "16", "3072", "4096"]).join(", "); }),
      mRaw = _b[0], setMRaw = _b[1];
  var _c = useState(function () { return (prev.nVals || ["1536"]).join(", "); }),
      nRaw = _c[0], setNRaw = _c[1];
  var _d = useState(function () { return (prev.kVals || ["4096"]).join(", "); }),
      kRaw = _d[0], setKRaw = _d[1];

  var emit = useCallback(function (m, n, k) {
    var mVals = parseCsvInts(m), nVals = parseCsvInts(n), kVals = parseCsvInts(k);
    if (!mVals.length || !nVals.length || !kVals.length) return;
    var lines = ["shapes:"], dimN = nVals[0], dimK = kVals[0];
    for (var i = 0; i < mVals.length; i++) {
      lines.push("  - {id: m" + mVals[i] + ", M: " + mVals[i] + ", N: " + dimN + ", K: " + dimK + "}");
    }
    onChange(lines.join("\n"));
  }, [onChange]);

  return html`
    <div class="dkao-guided-form">
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">M (rows)</label>
        <input type="text" class="input dkao-guided-input"
          value=${mRaw}
          placeholder="e.g. 2, 16, 3072, 4096"
          onInput=${function (e) { setMRaw(e.target.value); emit(e.target.value, nRaw, kRaw); }} />
        <span class="dkao-guided-hint">1 ≤ M ≤ 4096; M=3072/4096 covers large prefill</span>
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">N (cols)</label>
        <input type="text" class="input dkao-guided-input"
          value=${nRaw}
          placeholder="e.g. 1536"
          onInput=${function (e) { setNRaw(e.target.value); emit(mRaw, e.target.value, kRaw); }} />
        <span class="dkao-guided-hint">Output feature dim</span>
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">K (inner)</label>
        <input type="text" class="input dkao-guided-input"
          value=${kRaw}
          placeholder="e.g. 4096"
          onInput=${function (e) { setKRaw(e.target.value); emit(mRaw, nRaw, e.target.value); }} />
        <span class="dkao-guided-hint">Reduction dim</span>
      </div>
    </div>
  `;
}

function RmsNormGuided(_a) {
  var value = _a.value, onChange = _a.onChange;
  var prev = useMemo(function () { return parseShapeYaml(value || ""); }, [value]);
  var _b = useState(function () { return (prev.tVals || ["1", "128"]).join(", "); }),
      tRaw = _b[0], setTRaw = _b[1];
  var _c = useState(function () { return (prev.hVals || ["7168"]).join(", "); }),
      hRaw = _c[0], setHRaw = _c[1];

  var emit = useCallback(function (t, h) {
    var tVals = parseCsvInts(t), hVals = parseCsvInts(h);
    if (!tVals.length || !hVals.length) return;
    var lines = ["shapes:"], dimH = hVals[0];
    for (var i = 0; i < tVals.length; i++) {
      lines.push("  - {id: rmsnorm_t" + tVals[i] + ", T: " + tVals[i] + ", H: " + dimH + "}");
    }
    onChange(lines.join("\n"));
  }, [onChange]);

  return html`
    <div class="dkao-guided-form">
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">T (tokens)</label>
        <input type="text" class="input dkao-guided-input"
          value=${tRaw}
          placeholder="e.g. 1, 128"
          onInput=${function (e) { setTRaw(e.target.value); emit(e.target.value, hRaw); }} />
        <span class="dkao-guided-hint">T = batch_size × seq_len</span>
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">H (hidden dim)</label>
        <input type="text" class="input dkao-guided-input"
          value=${hRaw}
          placeholder="e.g. 7168"
          onInput=${function (e) { setHRaw(e.target.value); emit(tRaw, e.target.value); }} />
        <span class="dkao-guided-hint">The last dimension being normalized</span>
      </div>
    </div>
  `;
}

function PrefillAttnGuided(_a) {
  var value = _a.value, onChange = _a.onChange;
  var prev = useMemo(function () { return parseShapeYaml(value || ""); }, [value]);
  var _b = useState(function () { return (prev.tokensVals || ["128", "1024", "4096"]).join(", "); }),
      tokensRaw = _b[0], setTokensRaw = _b[1];
  var _c = useState(function () { return (prev.hdVals || ["128"]).join(", "); }),
      hdRaw = _c[0], setHdRaw = _c[1];
  var _d = useState(true), kvLenSame = _d[0], setKvLenSame = _d[1];
  var _e = useState(""), kvLenRaw = _e[0], setKvLenRaw = _e[1];

  var emit = useCallback(function (tokens, hd, same, kv) {
    var tVals = parseCsvInts(tokens), hdVals = parseCsvInts(hd);
    if (!tVals.length || !hdVals.length) return;
    var lines = ["shapes:"], dimHd = hdVals[0];
    for (var i = 0; i < tVals.length; i++) {
      var klen = same ? tVals[i] : (parseCsvInts(kv)[0] || tVals[i]);
      lines.push("  - {id: prefill_t" + tVals[i] + ", M: " + tVals[i] + ", N: " + klen + ", K: " + dimHd + "}");
    }
    onChange(lines.join("\n"));
  }, [onChange]);

  return html`
    <div class="dkao-guided-form">
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">total_tokens (M)</label>
        <input type="text" class="input dkao-guided-input"
          value=${tokensRaw}
          placeholder="e.g. 128, 1024, 4096"
          onInput=${function (e) { setTokensRaw(e.target.value); emit(e.target.value, hdRaw, kvLenSame, kvLenRaw); }} />
        <span class="dkao-guided-hint">M = batch × seq_len</span>
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">head_dim (K)</label>
        <input type="text" class="input dkao-guided-input"
          value=${hdRaw}
          placeholder="e.g. 128"
          onInput=${function (e) { setHdRaw(e.target.value); emit(tokensRaw, e.target.value, kvLenSame, kvLenRaw); }} />
        <span class="dkao-guided-hint">Head dimension, typically 128</span>
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-check">
          <input type="checkbox"
            checked=${kvLenSame}
            onClick=${function () { var next = !kvLenSame; setKvLenSame(next); emit(tokensRaw, hdRaw, next, kvLenRaw); }} />
          <span>kv_len (N) = total_tokens (self-attention)</span>
        </label>
      </div>
      ${!kvLenSame && html`
        <div class="dkao-guided-row">
          <label class="dkao-guided-label">kv_len (N)</label>
          <input type="text" class="input dkao-guided-input"
            value=${kvLenRaw}
            placeholder="e.g. 32768"
            onInput=${function (e) { setKvLenRaw(e.target.value); emit(tokensRaw, hdRaw, kvLenSame, e.target.value); }} />
          <span class="dkao-guided-hint">Cross-attention KV length</span>
        </div>
      `}
    </div>
  `;
}

function DecodeAttnGuided(_a) {
  var value = _a.value, onChange = _a.onChange;
  var prev = useMemo(function () { return parseShapeYaml(value || ""); }, [value]);
  var _b = useState(function () { return (prev.kvVals || ["4096", "32768"]).join(", "); }),
      kvRaw = _b[0], setKvRaw = _b[1];
  var _c = useState(function () { return (prev.hdVals || ["128"]).join(", "); }),
      hdRaw = _c[0], setHdRaw = _c[1];

  var emit = useCallback(function (kv, hd) {
    var kVals = parseCsvInts(kv), hdVals = parseCsvInts(hd);
    if (!kVals.length || !hdVals.length) return;
    var lines = ["shapes:"], dimHd = hdVals[0];
    for (var i = 0; i < kVals.length; i++) {
      lines.push("  - {id: decode_kv" + kVals[i] + ", M: 1, N: " + kVals[i] + ", K: " + dimHd + "}");
    }
    onChange(lines.join("\n"));
  }, [onChange]);

  return html`
    <div class="dkao-guided-form">
      <div class="dkao-guided-info">
        <strong>M = 1</strong> (fixed — decode processes 1 token at a time).
        Bottleneck is KV cache read; this is memory-bound.
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">kv_len (N)</label>
        <input type="text" class="input dkao-guided-input"
          value=${kvRaw}
          placeholder="e.g. 4096, 32768"
          onInput=${function (e) { setKvRaw(e.target.value); emit(e.target.value, hdRaw); }} />
        <span class="dkao-guided-hint">KV cache length — the decode bottleneck</span>
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">head_dim (K)</label>
        <input type="text" class="input dkao-guided-input"
          value=${hdRaw}
          placeholder="e.g. 128"
          onInput=${function (e) { setHdRaw(e.target.value); emit(kvRaw, e.target.value); }} />
        <span class="dkao-guided-hint">Head dimension, typically 128</span>
      </div>
    </div>
  `;
}

function RoPeGuided(_a) {
  var value = _a.value, onChange = _a.onChange;
  var prev = useMemo(function () { return parseShapeYaml(value || ""); }, [value]);
  var _b = useState(function () { return (prev.tVals || ["1", "128", "4096"]).join(", "); }),
      tRaw = _b[0], setTRaw = _b[1];
  var _c = useState(function () { return (prev.hdVals || ["128"]).join(", "); }),
      hdRaw = _c[0], setHdRaw = _c[1];

  var emit = useCallback(function (t, hd) {
    var tVals = parseCsvInts(t), hdVals = parseCsvInts(hd);
    if (!tVals.length || !hdVals.length) return;
    var lines = ["shapes:"], dimHd = hdVals[0];
    for (var i = 0; i < tVals.length; i++) {
      lines.push("  - {id: rope_t" + tVals[i] + ", T: " + tVals[i] + ", H: " + dimHd + "}");
    }
    onChange(lines.join("\n"));
  }, [onChange]);

  return html`
    <div class="dkao-guided-form">
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">T (tokens)</label>
        <input type="text" class="input dkao-guided-input"
          value=${tRaw}
          placeholder="e.g. 1, 128, 4096"
          onInput=${function (e) { setTRaw(e.target.value); emit(e.target.value, hdRaw); }} />
        <span class="dkao-guided-hint">T = batch × seq_len</span>
      </div>
      <div class="dkao-guided-row">
        <label class="dkao-guided-label">head_dim (H)</label>
        <input type="text" class="input dkao-guided-input"
          value=${hdRaw}
          placeholder="e.g. 64, 128"
          onInput=${function (e) { setHdRaw(e.target.value); emit(tRaw, e.target.value); }} />
        <span class="dkao-guided-hint">Per-head rotary dimension</span>
      </div>
    </div>
  `;
}

// ---- YAML parser (best-effort, for backfilling guided fields on toggle) -----

function parseShapeYaml(raw) {
  var out = {};
  if (!raw) return out;
  var mVals = [], nVals = [], kVals = [], tVals = [], hVals = [];
  var tokensVals = [], hdVals = [], kvVals = [];
  var re = /\{[^}]+\}/g, m;
  while ((m = re.exec(raw)) !== null) {
    var inner = m[0];
    var get = function (key) {
      var km = new RegExp(key + ":\\s*(\\d+)").exec(inner);
      return km ? parseInt(km[1], 10) : null;
    };
    var mv = get("M"); if (mv != null) mVals.push(mv);
    var nv = get("N"); if (nv != null) nVals.push(nv);
    var kv = get("K"); if (kv != null) kVals.push(kv);
    var tv = get("T"); if (tv != null) tVals.push(tv);
    var hv = get("H"); if (hv != null) hVals.push(hv);
    var toks = mv; if (toks != null) tokensVals.push(toks);
    var hd = kv; if (hd != null) hdVals.push(hd);
    var kvLen = nv; if (kvLen != null) kvVals.push(kvLen);
  }
  var uniqSort = function (arr) { return arr.filter(function (v, i) { return arr.indexOf(v) === i; }).sort(function (a, b) { return a - b; }); };
  if (mVals.length) out.mVals = uniqSort(mVals);
  if (nVals.length) out.nVals = uniqSort(nVals);
  if (kVals.length) out.kVals = uniqSort(kVals);
  if (tVals.length) out.tVals = uniqSort(tVals);
  if (hVals.length) out.hVals = uniqSort(hVals);
  if (tokensVals.length) out.tokensVals = uniqSort(tokensVals);
  if (hdVals.length) out.hdVals = uniqSort(hdVals);
  if (kvVals.length) out.kvVals = uniqSort(kvVals);
  return out;
}

// ---- operator → component map ----------------------------------------------

var OPERATOR_GUIDED = {
  "Quantized GEMM": GemmGuided,
  "Attention": PrefillAttnGuided,
  "RMSNorm / LayerNorm": RmsNormGuided,
  "RoPE": RoPeGuided,
};

// ---- main component ---------------------------------------------------------

function ShapeInputInner(_a) {
  var field = _a.field, value = _a.value, onChange = _a.onChange, allValues = _a.allValues;
  var operator = (allValues && allValues.operator) || "";
  var dtype = (allValues && allValues.dtype) || "";
  var model = (allValues && allValues.model) || "DeepSeek V4 Flash";
  var assignmentMode =
    (allValues && allValues.shape_assignment_mode) || "AI automatic";
  var manualAssignment = assignmentMode === "Manual by GPU";
  var scopeMode =
    (allValues && allValues.shape_scope) || "All API shapes";
  var subsetScope = scopeMode === "Selected shapes only";
  var supportsModelCatalog =
    operator === "Quantized GEMM" && dtype === "INT8 W8A8";
  var usesApiDefaults =
    supportsModelCatalog &&
    model === "DeepSeek V4 Flash" &&
    !(value && value.trim());
  var isCustom = operator === "Custom operator" || !OPERATOR_GUIDED[operator];

  var _b = useState(function () {
    if (usesApiDefaults) return "api";
    if (isCustom) return "raw";
    if (value && /\bassignments\s*:/.test(value)) return "raw";
    if (supportsModelCatalog && model !== "DeepSeek V4 Flash") {
      return "catalog";
    }
    return "guided";
  }), mode = _b[0], setMode = _b[1];

  var prevOpRef = useRef(operator);
  useEffect(function () {
    if (prevOpRef.current !== operator) {
      prevOpRef.current = operator;
      if (!isCustom) setMode("guided");
    }
  }, [operator, isCustom]);

  useEffect(function () {
    if (usesApiDefaults) {
      setMode("api");
    } else if (
      mode === "api" ||
      (mode === "catalog" && model === "DeepSeek V4 Flash")
    ) {
      setMode(
        isCustom
          ? "raw"
          : supportsModelCatalog && model !== "DeepSeek V4 Flash"
            ? "catalog"
            : "guided"
      );
    }
  }, [usesApiDefaults, isCustom, supportsModelCatalog, model]);

  var _c = useState(function () {
    if (operator === "Attention" && value) {
      if (/\bM:\s*1[,\s\}]/.test(value)) return "decode";
    }
    return "prefill";
  }), attnSubMode = _c[0], setAttnSubMode = _c[1];

  var GuidedCmp = OPERATOR_GUIDED[operator];

  useEffect(function () {
    if (
      !manualAssignment &&
      value &&
      /\bassignment_mode\s*:\s*manual\b/.test(value)
    ) {
      onChange("");
    }
  }, [manualAssignment]);

  useEffect(function () {
    if (
      !manualAssignment &&
      !subsetScope &&
      value &&
      /\bshape_scope\s*:\s*subset\b/.test(value)
    ) {
      onChange("");
    }
  }, [manualAssignment, subsetScope]);

  if (manualAssignment) {
    var supportsManualDefaults = supportsModelCatalog;
    return html`
      <div class="dkao-shape-input">
        <div class="dkao-shape-tabs">
          <button class="dkao-shape-tab active" type="button">
            Manual by GPU
          </button>
          <span class="dkao-shape-mode-note">
            ${supportsManualDefaults
              ? "Fixed API workload"
              : "Advanced YAML required"}
          </span>
        </div>
        <div class="dkao-shape-body">
          ${supportsManualDefaults
            ? html`<${ManualGpuAssignment}
                key=${model}
                value=${value}
                onChange=${onChange}
                scopeMode=${scopeMode}
                model=${model} />`
            : html`
              <div class="dkao-guided-info">
                This operator does not expose a fixed workload for the four-card
                editor yet. Define <code>shapes</code> and
                <code>assignments</code> explicitly below.
              </div>
              <textarea class="input" rows="10"
                value=${value != null ? value : ""}
                placeholder="assignment_mode: manual\nshapes:\n  - {id: shape0, M: 1, N: 1, K: 1}\nassignments:\n  worker_0: {gpu: 0, shapes: [shape0]}"
                onInput=${function (event) {
                  return onChange(event.target.value);
                }}></textarea>
            `}
        </div>
      </div>
    `;
  }

  if (
    subsetScope &&
    supportsModelCatalog
  ) {
    return html`
      <div class="dkao-shape-input">
        <div class="dkao-shape-tabs">
          <button class="dkao-shape-tab active" type="button">
            AI automatic · selected subset
          </button>
          <span class="dkao-shape-mode-note">Fixed API workload</span>
        </div>
        <div class="dkao-shape-body">
          <${AiShapeSubset}
            key=${model}
            value=${value}
            onChange=${onChange}
            model=${model} />
        </div>
      </div>
    `;
  }

  return html`
    <div class="dkao-shape-input">
      <div class="dkao-shape-tabs">
        ${usesApiDefaults && html`
          <button class="dkao-shape-tab active" onClick=${function () {}}>
            API defaults
          </button>
        `}
        ${supportsModelCatalog && model !== "DeepSeek V4 Flash" && html`
          <button
            class="dkao-shape-tab ${mode === "catalog" ? "active" : ""}"
            onClick=${function () { return setMode("catalog"); }}>
            Model workload
          </button>
        `}
        <button
          class="dkao-shape-tab ${mode === "guided" ? "active" : ""}"
          disabled=${isCustom}
          onClick=${function () { return setMode("guided"); }}
          title=${isCustom ? "Guided mode not available for Custom operator" : "Guided shape form"}>
          Guided
        </button>
        <button
          class="dkao-shape-tab ${mode === "raw" ? "active" : ""}"
          onClick=${function () { return setMode("raw"); }}>
          Raw
        </button>
        ${operator === "Attention" && mode === "guided" && html`
          <span class="dkao-shape-spacer"></span>
          <button
            class="dkao-shape-tab ${attnSubMode === "prefill" ? "active" : ""}"
            onClick=${function () { return setAttnSubMode("prefill"); }}>
            Prefill
          </button>
          <button
            class="dkao-shape-tab ${attnSubMode === "decode" ? "active" : ""}"
            onClick=${function () { return setAttnSubMode("decode"); }}>
            Decode
          </button>
        `}
      </div>
      <div class="dkao-shape-body">
        ${mode === "api"
          ? html`
            <div class="dkao-guided-info">
              <strong>No manual shape input required.</strong>
              MetaInfer will read <code>DEFAULT_OPTIMIZATION_SHAPES</code>
              from the immutable INT8 W8A8 GEMM API when the task starts.
              Choose Guided or Raw only to override that workload.
            </div>
            <${ApiDefaultsPreview} catalog=${modelCatalog(model)} />
          `
          : mode === "catalog"
          ? html`<${ModelWorkloadAll}
              value=${value}
              onChange=${onChange}
              model=${model} />`
          : mode === "raw" || !GuidedCmp
          ? html`
            <textarea class="input" rows="6"
              value=${value != null ? value : ""}
              placeholder=${field.help || ""}
              onInput=${function (e) { return onChange(e.target.value); }}></textarea>
          `
          : operator === "Attention" && attnSubMode === "decode"
            ? html`<${DecodeAttnGuided} value=${value} onChange=${onChange} />`
            : html`<${GuidedCmp} value=${value} onChange=${onChange} />`
        }
      </div>
    </div>
  `;
}

// ---- register ---------------------------------------------------------------

var _g = (typeof globalThis !== "undefined" ? globalThis : window);
var _bridge = (_g.__metainferOverrides = _g.__metainferOverrides || {});
_bridge["shape-input"] = ShapeInputInner;

// Named export for the form-renderer to pick up.
export var ShapeInput = ShapeInputInner;
