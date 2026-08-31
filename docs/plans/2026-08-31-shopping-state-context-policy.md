# ShoppingState + 固定 K 轮上下文策略：设计更迭报告

**状态：核心 Harness 已实施；未启动教师采集、SFT、GRPO 或 Final-200 评测。**

实施范围：已落地确定性 state reducer、极简历史占位符、固定 K=3 的教师/评测 rollout
视图、action-level SFT 前缀重放和 GRPO token-span 替换，并新增对应单元测试。尚未执行
Phase A 的 127 条离线容量 replay，也未开展 Phase C 教师试采集；两者仍是启动训练前的
必经验收步骤。

## 0. 本次要冻结的决策

将当前“超过 token 预算才清理旧 tool result”的策略替换为一个统一的、可重放的
上下文契约：`shopping-state-context-v1`。

```text
固定保留最近 K=3 个完整 assistant + tool 交互组
更早交互组保留 assistant tool call，但 tool result 固定替换为历史占位符
每次环境成功返回后，Harness 确定性更新一份只读 ShoppingState
下一次模型决策同时看到：任务 + ShoppingState + 历史占位符 + 最近 3 个完整结果
```

这里的 K=3 是**上下文消息历史**的保留规则，不是 `ShoppingState` 的字段。两者的
管辖范围不同，但不是互斥的：`ShoppingState` 从**截至当前的全部成功环境事件**归约
事实，因而也覆盖最近三轮；最近三轮的完整 tool result 则保留原始页面、正文、选项和
按钮。换言之，state 不复制这三轮的整页 observation，却会保留其中已经确认的跨轮事实
（例如已搜索 query、已查看 ASIN、最终选中规格与价格）。

模型 prompt 必须显式写明以下边界：

```text
[SHOPPING_STATE_V1] 是截至 as_of_step 的只读事实账本，覆盖所有此前成功执行的工具结果，
包括最近三轮；它不是当前页面，也不提供按钮。
当前 tool result 是唯一的可执行页面；只能对其中列出的按钮、ASIN 或规格执行工具调用。
最近三轮完整结果用于核对原始细节；若与 state 表述不一致，以当前页面为准，并以最新
as_of_step 的 state 作为历史事实的权威版本。
```

本报告定义已实施的设计、数据契约、验收门槛和后续执行顺序。

## 1. 为什么需要这项变更

### 1.1 问题

购物 Agent 会搜索、打开多个相似商品、查看详情与规格。若把全部原始 observation
无上限保留：

- 搜索页的 20 个商品、详情页与规格会持续累积；
- GRPO 同一任务要同时 rollout 多条轨迹并计算 log probability，显存随上下文长度显著上升；
- SFT 与 GRPO 容易出现超长样本、截断或 OOM；
- 直接截断字符串会丢掉 ASIN、当前按钮，导致模型与 Action Guard 对“当前可点目标”理解不一致。

只删除历史也不可取：模型会忘记已经搜索、查看和比较过的候选，造成重复搜索或重复打开。

### 1.2 当前实现与目标设计的差异

当前仓库已有 `clear_old_tool_results(..., keep_recent_groups=3)`，但它只在
`original_tokens > input_budget` 时运行；而且只有单条 observation 的通用占位符，没有
独立的候选、query 与规格状态。因此当前行为是：

```text
上下文较短：完整保留全部历史
超过预算：才清理早于最近 3 轮的 tool result
仍超过预算：message rollout 删除完整旧组；GRPO token rollout 可能终止
```

这与本次目标不同。目标行为从第 4 个成功环境结果开始就稳定，不依赖当时 token
长度：

```text
任何长度：早于最近 3 个完整交互组的 tool result 都是占位符
任何长度：最新 ShoppingState 都可见
接近硬上限：只触发预定义的安全失败或经过验收的二级缩减；不静默换策略
```

### 1.3 实证容量依据

对当前已完成的 127 条教师原始轨迹统计：

| 轨迹行为 | 严格 gold P95 / 最大值 | 全部轨迹 P95 / 最大值 |
|---|---:|---:|
| 不同 query 数 | 5 / 12 | 10 / 12 |
| 打开不同商品数 | 4 / 5 | 5 / 9 |
| 证据页总数 | 4 / 7 | 6 / 10 |
| 单商品查看证据页数 | 3 / 3 | 3 / 3 |
| `select_option` 调用数 | 3 / 6 | 5 / 7 |

这 127 条是**未压缩**采集的审计/基线数据，不能混入新策略的训练集；但其交互规模
可以作为 state 容量的初始依据。

## 2. 设计目标与非目标

### 2.1 目标

1. 教师采集、action-level SFT、GRPO rollout、线上评测对同一事件前缀构造相同的模型输入。
2. 当前页的 observation 仍是唯一可执行动作边界；记忆中的旧 ASIN、旧规格绝不能变成可点击目标。
3. 所有记忆由代码从 agent-visible action/observation 更新，不调用额外 LLM，不使用 Reward、gold 商品、预算标签或用户 persona 私有字段。
4. 每一轮 state、占位、裁剪决定可审计、可重放、可单元测试。
5. 对超长轨迹给出稳定的长度上限，而不是依赖字符串截断。

### 2.2 非目标

- 不新增“思考”“总结”或 LLM memory 工具。
- 不让 Harness 推测模型主观的“拒绝原因”。
- 不把全部搜索页的候选和完整详情镜像进 state。
- 不修改 ShopSimulator 商品、任务或 Reward v3 的 gold 定义。
- 本变更不把 127 条未压缩教师轨迹伪装成压缩 Harness 数据。

## 3. `ShoppingState` 数据契约

### 3.1 状态内容

```json
{
  "version": "shopping-state-v1",
  "as_of_step": 8,
  "searched_queries": [
    {"query": "自动浇水器 铜电磁阀", "result_count": 150}
  ],
  "reviewed_products": [
    {
      "asin": "750684323117",
      "title": "自动浇水器铜电磁阀浇水器…",
      "category": "…自动灌溉设备",
      "price": "85.0 to 298.0",
      "key_attributes": ["自动浇水器", "电磁阀", "雾化"],
      "viewed_pages": ["description"],
      "selected_options": {"颜色分类": "…25米快插式地插喷头套装"},
      "selected_price": 228.0,
      "selection_status": "current_candidate"
    }
  ],
  "reviewed_product_archive": [
    {"asin": "677758868630", "title": "大屏幕双路控制自动浇水器…", "price": "276.0 to 778.0", "selection_status": "reviewed_not_selected"}
  ],
  "current_product_asin": "750684323117"
}
```

字段均为 observation 中已出现的事实。state 的**语义范围**是所有已成功执行的历史步骤，
包含最近三轮；其**容量范围**是受上限约束的事实账本，而非最近三轮 observation 的副本。
`asin` 明确标为历史只读标识，不渲染为按钮，Action Guard 仍只检查当前页 observation。

### 3.2 不能臆造“排除原因”

模型从详情页返回搜索页，并不等于它明确表示“该商品因价格超预算被排除”。第一版仅
记录 `reviewed_not_selected` 与已观察的价格、属性、规格事实。

只有可由 agent-visible 内容确定的事实才可写入 state，例如“已选规格后的实际价格
为 228.0”。若未来确实要训练“排除原因”，应单独设计可审计的
`reject_candidate(reason)` 工具；它不属于本次变更。

### 3.3 更新规则

| 成功执行事件 | 归约器更新 |
|---|---|
| `search_products` | 去重追加 query；记录环境返回的结果总数；不保存整页商品列表。 |
| `open_product` | 新建或刷新商品卡片：ASIN、短标题、类目、价格、结构化属性；设为 `under_review` 与当前商品。 |
| `view_description/features/reviews/attributes` | 给当前商品追加已查看页面；最多保留受限长度、原文可追溯的 evidence excerpt。 |
| `select_option` | 从**选择后** observation 更新规格轴最终值与确定价格；同一轴只保留当前生效值。 |
| `prev_page/back_to_search/next_page` | 不删除候选事实；离开详情页后将非当前候选记为 `reviewed_not_selected`。 |
| `buy_now` / `finish_without_purchase` | 标记终局，不再构造下一轮 Agent 输入。 |
| Guard 拒绝 / 解析失败 / 环境失败 | 不改变 state；仅写入轨迹审计。 |

归约器必须读取原始、未投影 observation，再生成 state；不得根据 reward/terminal gold
字段回填状态。

### 3.4 初始容量与确定性淘汰

基于第 1.3 节统计，`v1` 的暂定上限是：

| 区域 | 上限 | 原因 |
|---|---:|---|
| `searched_queries` | 12 | 覆盖当前样本最大值。 |
| 完整 `reviewed_products` 卡片 | 8 | 覆盖所有 strict-gold 轨迹；127 条中仅 1 条打开 9 个商品。 |
| 轻量 `reviewed_product_archive` | 12 | 保存溢出候选的 ASIN、短标题、价格与状态。 |
| 每商品 `viewed_pages` | 3 | 当前样本最大值为 3。 |
| evidence excerpts | 全局 6 条，每条 160 字符 | P95 全部轨迹为 6 页；避免详情正文膨胀。 |
| 当前候选 | 1，永不淘汰 | 当前商品是下一步决策的关键。 |
| 每规格轴值 | 1 个最终生效值 | 不保留连续试选的历史值。 |

淘汰优先级固定为：保护当前候选与有已选规格的卡片；其余按最早最后访问时间从完整
卡片降级到轻量 archive；archive 满时删除最早的非当前条目。相同时间以 ASIN 字典序
打破平局。不得按模型概率或 Reward 选择淘汰对象。

这些是结构容量，并非最终 token 预算。实施前必须在 127 条原始轨迹上离线 replay，
用 Qwen tokenizer 验证序列化 state 的 P95 不超过 2,048 tokens，并记录最大值；若
超出，优先缩短标题与证据摘录，而不是减少候选卡片数。

## 4. 模型实际看到的上下文

对第 `t` 次模型决策，令 `G_i = assistant tool-call_i + tool result_i` 为完整交互组，
`S_t` 为在第 `t-1` 次环境成功返回后得到的 state。

```text
固定 system + 用户任务
[SHOPPING_STATE_V1 as_of_step=t-1, read_only=true]
G_1 ... G_(t-4)：保留 assistant tool-call；tool result 改为极简非行动历史占位符
G_(t-3) ... G_(t-1)：保留完整 tool result
→ 生成 action_t
```

当历史不足 3 组时，所有现有 tool result 都完整保留。

历史占位符**不保存商品事实**。工具名和调用参数已保留在其前一个 assistant tool-call 中；
跨轮事实则唯一由 `ShoppingState` 保存。占位符只显式声明：

```text
该工具结果已从活跃上下文中清除。
这里没有按钮、页面内容或可执行目标。
历史事实请读取 [SHOPPING_STATE_V1]；只有当前 tool result 定义合法动作。
```

目标渲染固定如下；它发生在一次结果落到“最近 3 组”之外时：

```text
[SHOPPING_TOOL_RESULT_CLEARED_V1]
Historical tool result removed from active context.
No buttons, page content, or actionable targets are available here.
Use [SHOPPING_STATE_V1] for historical facts; only the current tool result defines legal actions.
```

例如，task 53 早期 `select_option` 的原始结果曾包含 ASIN `750684323117`、价格 `228.0`
及已选规格；这些事实会进入 state，而落到历史区后该 tool message 本身只变为上述三行。
它不保留 `available_options`、搜索结果列表、`可点击的按钮`、详情正文，也不重复 ASIN、
价格或规格。

`ShoppingState` 放入最新模型可见 observation 的无按钮前缀，标识为
`[SHOPPING_STATE_V1]`。最近三轮中可能同时存在较早 state snapshot，但每一份都带
`as_of_step`，最新 tool result 中的 state 是权威版本。Action Guard 接收单独保存的
`current_page_observation`，而不是解析 state 增强后的文本。

## 5. 固定 K 策略与长度策略

K=3 清理**不依赖 tokenizer**：第 4 个成功交互组开始固定发生。教师采集只构造这份
确定的文本，然后直接发送给远端 DeepSeek；不调用本地学生模型的 `/tokenize`，也不以
token 数决定是否清理。这避免额外 RPC、吞吐下降以及“教师采集和实际 Harness 规则不同”。

经源码核对，当前 `collect_sft_data.py` / `collection/sft.py` 的教师轨迹筛选只检查
strict-gold、Guard、终局与评测集隔离，**没有**“整条轨迹 token 太长则跳过”的条件。
长度筛选发生在后续 `build_action_supervised_examples`：某个 action 的 Qwen-chat-template
前缀超过 SFT `--max-length`，该 action 样本被丢弃，原始 trajectory 仍留在审计数据中。
因此 v1 应把这个事实显式记录为 `sft_dropped_over_max_length` 统计，而不是误称为教师
采集时已经跳过长轨迹。

Qwen tokenizer 只在**不位于教师采集关键路径**的两个位置使用：

1. Phase A 对已采集 raw trajectory 离线 replay，测量 state、占位符和 action prefix 的
   token 分布，确定本地学生训练长度；
2. SFT tokenization、GRPO rollout 与本地评测中，为 Qwen 的真实上下文窗口做硬安全检查。

暂定学生窗口为 24,576 tokens，单轮输出预留 512、安全余量 512，理论硬输入上限为
23,552。该上限不是教师采集的压缩触发器。若远端教师自身拒绝极长请求，采集将记录该
次 transport / provider failure，不静默改变 K=3 规则；若学生阶段某 action 前缀超过
冻结长度，则按上述 action-level 规则剔除并报告。

`v1` 不静默执行“删除更多完整历史组”的二级策略。学生运行时若“state + 所有历史占位符 +
最近三组完整结果”超过其冻结窗口，应明确标记 `context_policy_budget_exhausted` 并停止
该 rollout；先用离线分布决定是否需要 `v2` 二级缩减策略。

## 6. 数据与运行时一致性

### 6.1 不能只修改教师 rollout

当前 `OpenAIChatClient`、SFT action dataset、GRPO adapter 分别实现自己的预算触发
清理。新策略必须把“状态归约 + 固定 K 视图构造”抽为单一纯函数，三处共同调用：

```text
raw action/observation event stream
        ↓
ShoppingState reducer
        ↓
build_context_view(events, state, k=3)
        ↓
teacher rollout / SFT action prefix / GRPO token rollout / evaluation
```

这里的“一致性”是**结构文本一致性**：四条路径对同一事件前缀都得到相同 state、相同
占位符和相同最近三组完整结果；不是要求教师采集同步调用 Qwen tokenizer，也不是以同一
token 预算触发压缩。

建议新增：

```text
src/shopping_grpo/environment/shopping_state.py
  - ShoppingState 数据类与 canonical JSON serializer
  - reduce_shopping_state(previous_state, action, raw_observation)
  - build_context_view(event_prefix, state_snapshot, keep_recent_groups=3)
  - 只读 state/占位符渲染与容量淘汰
```

现有 `context.py` 的 `tool_result_placeholder` 改为极简清除标记，不再提取或复制 observation
字段，也不再由“是否超预算”决定是否调用。

### 6.2 原始审计与训练视图必须分离

不能覆盖 `raw.jsonl` 里的完整环境 observation；否则无法复核 state 是否从真实页面
归约而来。

每条新 raw trajectory 至少新增：

```text
context_policy_version: shopping-state-context-v1
state_trace: 每个成功环境步骤后的 canonical state snapshot
context_trace: 每个 assistant action 前的 state snapshot、K=3 清理计数与 canonical context hash
```

Qwen token 数是 Phase A/SFT/GRPO/评测的派生测量，不写入教师采集的每轮关键路径；如需
审计，可在 raw trajectory 完结后离线生成独立的 token-distribution report。

`raw_messages` 保留完整 audit 事件；action-level SFT 不应简单复用最终时刻被全局改写
过的消息串，而应对每个 assistant action 的前缀调用同一个纯函数重建当时模型看到的
context。这样第 1 个 action 不会错误看到“未来才发生的占位替换”。

### 6.3 SFT

当前 `build_action_supervised_examples` 只在 prefix 超预算时调用
`clear_old_tool_results`。实施后它必须：

1. 读取 raw event prefix 与对应 `state_trace`；
2. 无条件对早于 K=3 的 tool result 应用占位符；
3. 将该 turn 的 `[SHOPPING_STATE_V1]` 视图加入 prefix；
4. 只对当前 assistant tool call 计算 loss；
5. 记录 `context_policy_version`、state token 数与历史占位数量。

旧的 `--result-clearing` 训练开关不能与新策略同时生效；应改为显式
`--context-policy shopping-state-v1`，并在 preflight 中拒绝混用。

### 6.4 GRPO

当前 GRPO adapter 在 token 超预算时替换旧 tool response spans。实施后应在每个成功
环境 tool response 后更新 state，并把状态作为该 response 的无按钮前缀；当完整组数
超过 3 时，无条件将旧 response span 替换为与 message rollout 相同的占位 token。

替换必须继续保持 `prompt_ids`、`response_mask`、`response_logprobs` 对齐。若 veRL
的 routed-expert 状态不支持安全替换，维持当前“基础设施无效并终止”的保守处理。

### 6.5 评测与线上 rollout

`evaluation/rollout.py` 是教师采集与离线评测的共同路径，必须使用同一个 context
builder。评测输出需要额外写出 context/state trace，方便区分“模型选错”与“状态归约
遗漏事实”。

## 7. 不泄漏与动作安全约束

1. reducer 输入只能是用户任务、成功执行的工具参数、agent-visible 原始 observation；禁止读 goal、gold ASIN、Reward detail、预算 sidecar、persona 或终局字段。
2. state 和占位符都不得产生“可点击的按钮”文本；state 中旧 ASIN 是只读历史标识，
   极简占位符不含 ASIN。
3. Action Guard 只依据 `current_page_observation` 判定，不因 state 中含旧 ASIN 放行。
4. state 只存事实，不存模型 hidden reasoning、链式思考或自由文本总结。
5. Guard 拒绝动作不写入 state，但保留于审计 trace；SFT 继续删除被拒绝的 assistant call 及其 error tool message。

## 8. 实施顺序与验收门槛

### Phase A：离线归约器与分布审计（不调用模型）

1. 实现 state reducer 与 context builder 的纯函数。
2. 用 127 条 raw trajectory replay，生成状态容量、token 分布、淘汰事件报告；这一步
   可以使用本地 Qwen tokenizer，但不在教师采集时执行。
3. 验证无 state 泄漏、无历史按钮、K=3 恰好生效。
4. 冻结 token 操作预算和 state 字段/容量。

**通过门槛：** 所有 replay 可重现；state P95 ≤ 2,048 Qwen tokens；所有最新页按钮和
ASIN 与原 observation 一致；旧 state/placeholder 无可点击目标。

### Phase B：四路径一致性测试（不采集新训练数据）

1. Message rollout、SFT action prefix、GRPO token rollout 对相同合成事件前缀产生等价文本/token 视图。
2. 覆盖搜索页、详情页、子页、选规格、Guard 拒绝、超过 3 组、容量淘汰与硬预算失败。
3. 运行真实本地 Stage-C 压缩 smoke trajectory，生成一份人类可读报告。

**通过门槛：** 单元/集成测试通过；原始 action sequence 不因 state 前缀改变 Guard
合法性；SFT response mask 与 GRPO logprob 对齐检查通过。

### Phase C：教师压缩试采集

1. 远端 `deepseek-v4-flash` 只用于 completion。每个成功组后无条件应用固定 K=3
   视图；采集过程不调用本地 Qwen tokenizer，也不按 token 数改写历史。
2. 采集 100 个非评测任务，配置必须显式写 `context_policy_version`。
3. 审阅至少 5 条发生占位的严格 gold 轨迹和 5 条长/失败轨迹。

**通过门槛：** 零 context-policy 基础设施错误；每条第 4 个成功组后都满足 K=3；严格
gold 率不低于未压缩 100 条试运行的 60% 减 10 个百分点（即至少 50%，仅作试采集
健康阈值，不作为模型排名）；无 Final-200 task ID 重叠。

### Phase D：正式数据、SFT、GRPO

仅在 Phase C 审阅通过后，重新采集目标训练规模；新数据单独命名为
`shopping-state-context-v1`。完成 SFT 后先运行小规模 smoke/开发集，再由用户明确
授权启动 GRPO 和 Final-200。

## 9. 回滚与数据处置

- 已停止的 127 条未压缩教师轨迹保留在
  `outputs/sft-collection-price-guard-v2/`，只作 baseline、容量统计和审计；不得并入
  新策略训练数据。
- 现有 SFT checkpoint、Final-200 结果不覆盖；它们的 context-policy 版本为旧配置。
- 若 Phase A/B/C 任一门槛失败，代码回退到本变更前的 Git commit，并不启动正式采集。
- 新旧数据、checkpoint、评测报告必须记录明确的 `context_policy_version`，禁止合并
  汇总指标。

## 10. 本次确认需要用户冻结的参数

确认后实施时，以下值将作为 `v1` 契约写进配置和 preflight：

```text
keep_recent_complete_groups = 3
searched_queries_cap = 12
reviewed_product_cards_cap = 8
reviewed_product_archive_cap = 12
evidence_excerpt_cap = 6 x 160 characters
state_token_budget = 2048 (须经 Phase A replay 验证)
student_context_window = 24576
student_generation_reserve = 512
student_safety_margin = 512
```

“操作输入预算”不在本报告中武断冻结：它必须由 Phase A 的实际 Qwen token 分布确定，
且不得低于“固定 prompt + state P95 + 最近三组完整 observation P95 + 历史占位符”的
总和。
