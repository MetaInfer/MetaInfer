# Sub-agents 模板（Inference）

这些模板对应 `curosr/skills/README.md` 中的 5 类角色，可直接在 Agent 对话中调用：

- `/contract-checker`
- `/ref-tracer`
- `/tdd-test-writer`
- `/impl-coder`
- `/integration-verifier`

## 推荐调用顺序

1. `/contract-checker`：先产出 shape/接口检查清单  
2. `/ref-tracer`：抽取最小参考实现路径  
3. `/tdd-test-writer`：先写测试用例  
4. `/impl-coder`：按测试实现  
5. `/integration-verifier`：端到端验证与日志审计

## 约束

- 必须以 `curosr/skills/inference_blueprint.json` 为唯一契约来源。
- 必须遵守 `curosr/skills/AGENT_SKILL.md` 的 TDD 与门禁规则。
- 组件测试未通过时，禁止跨组件推进。
