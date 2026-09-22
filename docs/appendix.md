# 附录：Q-AVS 核心方法与实现细节

## A. 来源与适用范围

本附录以2026-09-07 的本地方法稿为暂定方法依据，并核对较新搜索实现中的提示、接受规则与参数定义。当前 Overleaf 主稿尚未取得，因此本文不能作为与其最新版本完全一致的证明。

本附录描述一次独立的 Q-AVS 搜索：输入高分辨率图像、问题和候选选项，输出被接受的答案或预先保存的全图答案。证据验证器是该推理算法的必要组成部分，负责检查视觉证据；它不使用标准答案。

以下代码位置采用整理后的 `qavs/` 包命名。参数已核对 `qavs/defaults.json` 与 `StrictNoConfig`；模型 checkpoint 由调用方固定指定。

## B. 输入、模型与搜索状态

记输入为图像 $I$、问题 $q$、语义选项集 $O$。答案生成器 $\mathcal M_g$ 与证据验证器 $\mathcal M_v$ 均保持冻结，且使用不同 checkpoint。模型适配器须提供答案生成及条件标签损失接口。

选项目录为每个语义答案建立稳定标识与输出映射。验证始终覆盖同一个完整选项集；异常或无法唯一映射的输出不能成为已接受答案。

搜索先冻结全图答案 $y^{\mathrm{global}}$。一次局部状态记录焦点区域、原图坐标、渲染尺度、根到焦点路径、上下文锚点、已访问候选及剩余预算。观察历史保留已访问状态，参与评分的证据则随当前路径重新筛选。

预算包含观察步数、模型调用量和处理像素量。通过接受门后立即停止；搜索结束仍无已接受答案时返回 $y^{\mathrm{global}}$。回退输出可能是 `No`，但回退本身不表示通过了否定答案的证据门。

## C. 问题规划、候选与动作

Query planner 保留原问题，生成不包含候选答案的定位表达、必要证据项，以及细节需求和上下文需求。实现提示要求四至六个定位短语，最多两个结构化证据项；证据项形式为目标细节、关系上下文或全局范围覆盖。

SAM 3 提供初始空间候选，SGAP 按需补充语义区域。全图保留为根节点；初始候选为空、覆盖不足或候选耗尽时可进入 `SCAN`。同一搜索中的 SGAP 构建可以复用。

候选优先级为

$$
R(c)=w_s S_q(c)+(1-w_s)[\lambda D_f(c)+(1-\lambda)D_e(c)].
$$

其中 $S_q$ 融合原问题和定位表达的 CLIP 相似度，$D_f$ 表示特征离散程度，$D_e$ 表示边缘信息。实现先对原问题分数、扩展表达分数、特征量和边缘量分别作候选组内百分位归一化，再作加权融合。

扩展表达分数取当前区域得分最高的至多 `top_k_augmented` 个表达的平均值；没有扩展表达时使用原问题分数。百分位采用并列平均秩，只有一个候选或所有分数相同时取 $0.5$。排序仅调整访问顺序。

| 动作 | 状态更新 |
| --- | --- |
| `ZOOM` | 围绕目标实例或当前焦点收紧原图裁剪，再重采样以增加目标细节。 |
| `SPLIT` | 进入当前区域中排名靠前的更细语义子区域。 |
| `EXPAND` | 加入空间邻近的上下文，使相关对象能够共同参与观察。 |
| `NEXT` | 访问下一个保留候选，并重新确定当前证据集合。 |
| `RECOVER` | 恢复仍有候选的状态，或触发覆盖扩展。 |
| `BACKTRACK` / `SCAN` | 分别实现状态回退与 SGAP 候选补充。 |

缺口评分提示只接收当前视觉输入、问题和必要证据描述，输出 `zoom`、`split`、`expand`、`next` 四个独立的 $[0,1]$ 数值。控制器结合问题需求、可执行动作及预算选择后续观察。

支持下降、反复证据不足、明确反驳和答案冲突可触发恢复。支持变化用于控制搜索方向；局部停止由下面的接受门决定。

## D. 独立验证与目标定位

### D.1 逐选项三标签验证

对视图 $e_j$ 和每个 $a\in O$，验证器评分标签集合
$\mathcal Z=\{\texttt{Support},\texttt{Refute},\texttt{Insufficient}\}$。
代码实际计算提示中 `A`、`B`、`C` 三个输出代码的条件损失，再按固定次序映射为语义标签。

$$
r_j(z\mid a)=\frac{\exp[-\ell_j(z\mid a)]}{\sum_{z'\in\mathcal Z}\exp[-\ell_j(z'\mid a)]},
\quad u_j(a)=r_j(\texttt{Support}\mid a),
\quad p_j(a)=\frac{u_j(a)}{\sum_{a'\in O}u_j(a')}.
$$

$u_j$ 保留绝对支持强度，$p_j$ 比较选项间的相对支持。有效验证要求全部选项具有三个完整有限损失、归一化有效、支持总和为正、生成与验证 checkpoint 不同，且至少一个选项的获胜标签不是 `Insufficient`。

固定选项验证提示为：

```text
Assess only direct visible evidence in this image; do not answer with another option and do not infer missing evidence. Classify with exactly one code: A = Support, B = Refute, C = Insufficient. Return only A, B, or C.
Question: {question}
Candidate answer: {option}
Required visible evidence:
{requirements}
```

### D.2 独立 Query grounding

对问题中的每个无答案角色 $r$，使用以下固定提示；该提示不包含候选答案：

```text
Decide whether this image directly and spatially grounds the named target role from the question. Do not answer the question and do not use candidate answers. Classify with exactly one code: A = Grounded, B = NotGrounded, C = Insufficient. Return only A, B, or C.
Question: {question}
Target role: {role}
```

将三个代码的条件损失作相同的负损失 softmax，得到 $\gamma_j(z\mid r)$。角色通过定位的条件为

$$
G_j(r)=[\arg\max_z\gamma_j(z\mid r)=\texttt{Grounded}]
\land[\gamma_j(\texttt{Grounded}\mid r)\ge\tau_g].
$$

通过独立定位的角色还须关联到稳定空间实例。实例来自 SAM 候选及其继承关系；必要时局部补充定位，按实例 IoU 阈值合并。SAM 标签负责空间实例关联，不能替代上述独立定位条件。

有效局部视图至少定位一个角色，且每个计入的角色均有实例标识。多目标问题允许一个视图仅贡献部分角色，在证据包层面再检查角色是否齐备。

## E. 证据组织与局部接受

### E.1 单目标支持度聚合

记 $\pi_j,C_j$ 为历史视图的路径与上下文，$\pi_t,C_t$ 为当前状态。准入规则为

$$
\mathrm{Eligible}_t(e_j)=\mathrm{Valid}_j^v\land\mathrm{Valid}_j^g
\land\mathrm{Ground}_j\land[\pi_j\preceq\pi_t]\land[C_j\subseteq C_t].
$$

其中 $\mathrm{Valid}^g$ 要求生成答案可合法映射到选项集。局部证据按来源图像、有效几何、渲染尺度、动作、内容哈希和路径去重；全图仅在通过严格验证后可作为共享证据。

对非空评分集合 $\mathcal E_t$，所有选项使用相同证据：

$$
U_t(a)=\frac1{|\mathcal E_t|}\sum_{e_j\in\mathcal E_t}u_j(a),
\qquad A_t(a)=\frac1{|\mathcal E_t|}\sum_{e_j\in\mathcal E_t}p_j(a).
$$

分支切换后立即重新筛选；失去路径或上下文匹配的记录留在历史中，不再贡献当前支持。

### E.2 多目标联合验证与共识配置

关系和比较问题要求证据角色并集覆盖全部必要角色。计数问题要求稳定实例去重，并包含覆盖所问范围的上下文。满足条件的局部视图与全图上下文形成确定性证据包 $B_t$。

联合配置在 $B_t$ 上重新生成答案并重新验证全部选项，取 $U_t=u_{B_t}$、$A_t=p_{B_t}$。各分支旧分数不在联合阶段平均；包上的生成答案也不单独触发停止。

预设的 grounded answer consensus 配置统计通过定位、生成答案与验证首选一致且相对支持比例达标的观察。属性问题按同一角色和实例归组；关系、比较和计数满足相应角色或实例条件。其接受分数来自有效确认比例及领先间隔，不能解释为验证器的正确率。

### E.3 接受门和不同观察

对支持聚合或联合验证，令 $\hat y_t=\arg\max_a A_t(a)$，$a_t^{(2)}$ 为第二名：

$$
\mathrm{Accept}_t=\mathrm{Valid}_t
\land[U_t(\hat y_t)\ge\tau_{\rm abs}]
\land[A_t(\hat y_t)-A_t(a_t^{(2)})\ge\tau_{\rm margin}]
\land\mathrm{DiverseConfirm}_t(\hat y_t).
$$

可确认答案 $a$ 的局部证据须以 `Support` 为获胜标签，且 $u_j(a)\ge\tau_{\rm view}$。至少两份确认须满足题目角色和实例条件，并存在一对不同观察：

$$
\mathrm{Different}(e_j,e_k)=[h_j^{\rm img}\ne h_k^{\rm img}]
\land[(b_j\ne b_k)\lor(s_j\ne s_k)\lor(d_j\ne d_k)].
$$

这里 $b,s,d,h^{\rm img}$ 分别是有效原图裁剪框、渲染尺度、动作和内容哈希。裁剪框按精确记录比较，不另设最小 IoU 差异；相同内容的重复输入不增加确认数，全图根节点也不计入局部确认。

单目标确认须指向同角色、同实例；关系和比较确认的角色并集须齐备；联合计数确认至少覆盖两个去重实例，并具有范围上下文。共识配置同样至少需要两个满足差异及目标条件的确认。

## F. 全图检查与二分类否定门

严格全图检查要求信息充分性得分严格大于 `direct_threshold`，生成输出有效并与验证首选一致，且绝对支持、相对间隔、生成一致性分别达到 `global_gate` 中的阈值。各阈值由运行配置给定。

默认启用支持度快捷门：验证有效、首选获胜标签为 `Support`、绝对支持与相对间隔达到对应阈值即可直接采用验证首选；该门不要求生成器答案一致。二分类存在性问题只允许快捷接受 `Yes`，也禁止通过严格全图检查直接接受 `No`。

记全图首选与次选为 $a_0^*,a_0^{(2)}$，则存在性问题的快捷门为

$$
\mathrm{FastAccept}_0=\mathrm{Valid}_0^v\land[a_0^*=\texttt{Yes}]
\land[z_0^*(a_0^*)=\texttt{Support}]\land[u_0(a_0^*)\ge0.65]
\land[p_0(a_0^*)-p_0(a_0^{(2)})\ge0.15].
$$

其他选择题使用相同规则，去掉首选必须是 `Yes` 的条件。

当局部门选择 `No` 时，还需检查 `StrictNoConfig`：全部相关候选已经访问，非全图观察的有效裁剪并集达到空间覆盖阈值；至少两个不同裁剪几何均强烈反驳 `Yes`；反驳证据包含 `GLOBAL` 或 `EXPAND` 上下文；所有有效视图对 `Yes` 的最大支持低于规定上限。

覆盖比例按裁剪框与原图边界相交后的并集面积除以原图面积计算，避免重叠区域重复累计。每次否定判断仅使用同一时刻已获得的证据与覆盖情况；后续覆盖不能追溯认可早先的否定答案。否定门的局部记录须有有效 grounding；全图上下文可贡献反驳。否定门未通过时继续搜索，最终无接受结果则使用全图回退。

## G. 参数与代码位置

| 参数 | 数值 | 依据 |
| --- | --- | --- |
| $\tau_g$ | $0.65$ | `defaults.json`：`verification.grounding_threshold`；与本地稿一致。 |
| 实例合并 IoU | $0.50$ | `verification.target_instance_iou`；与本地稿一致。 |
| $\tau_{\rm abs},\tau_{\rm margin},\tau_{\rm view}$ | $0.65,0.15,0.60$ | `acceptance`；与本地稿效率配置一致。 |
| 快捷门支持、间隔 | $0.65,0.15$ | `acceptance.global_verifier_min_support`、`global_verifier_min_margin`。 |
| 默认聚合方式 | `global_verifier_then_branch_equal_mean` | 先检查全图快捷门，再采用分支支持度聚合。 |
| 严格全图充分性、支持、间隔、一致性 | $>0.80,\ge0.80,\ge0.20,\ge0.75$ | `direct_threshold` 与 `global_gate`。 |
| 排序语义权重、主问题融合、视觉融合 | $0.60,0.50,0.50$ | `ranking.alpha`、`beta`、`visual_lambda`。 |
| 扩展定位表达 Top-$k$ | $3$ | `ranking.top_k_augmented`。 |
| 观察步数、模型调用量、处理像素量 | $8,256,1{,}600{,}000{,}000$ | `budget.max_steps`、`max_model_calls`、`max_processed_pixels`。 |
| SAM 去重 IoU、最低候选数、最低空间覆盖 | $0.90,1,0.005$ | `proposals.dedup_iou`、`min_count`、`min_spatial_coverage`。 |
| 活动候选数 | $4$ | `proposals.active_top_k`；其他候选保留于候选池。 |
| 最小缩放因子、最大归一化上下文间隔 | $0.40,0.25$ | `observation_geometry`。 |
| 停滞耐心、最小进展 | $2,0.02$ | `controller.stall_patience`、`min_progress`。 |
| 否定空间覆盖 | $\ge0.80$ | `StrictNoConfig.min_coverage` 默认值。 |
| 对 `Yes` 的反驳概率 | $\ge0.70$ | `StrictNoConfig.min_refute_probability` 默认值。 |
| 对 `Yes` 的最大支持 | $<0.70$ | `StrictNoConfig.max_yes_support_exclusive` 默认值。 |
| 不同反驳几何数 | $\ge2$ | `StrictNoConfig.min_distinct_refute_views` 默认值。 |

配置中的其他控制器字段不替代 E.3 的接受门；核心适配明确关闭旧控制器的 `certified_stop`。本地方法稿的效率模型组合为 InternVL2.5-8B 生成器和 LLaVA-OV-7B 验证器，具体 checkpoint 及身份由运行调用提供。

| 模块或提示 | 整理后代码位置 |
| --- | --- |
| 单次搜索、全图回退、恢复与覆盖计算 | `qavs/independent_search/method.py`：`run_independent_sample`、`_negative_candidate_coverage`。 |
| 配置与预算 | `qavs/independent_search/config.py`；`qavs/evidence_gap/pdf_types.py`。 |
| SAM 候选及 SGAP 恢复 | `qavs/independent_search/frontend.py`；`qavs/models/modeling_sam3.py`、`tree.py`。 |
| 排序与 CLIP | `qavs/evidence_gap/ranking.py`；`qavs/evidence_gap/clip_scorer.py`。 |
| Query planner 与缺口评分的完整固定提示 | `qavs/evidence_gap/pdf_runtime.py`：`_PLAN_PROMPT`、`_GAP_PROMPT`。 |
| 逐选项验证提示与支持向量 | `qavs/independent_search/semantics.py`：`_OPTION_VERIFIER_PROMPT`、`verify_option_support`。 |
| 独立定位提示及实例关联 | `qavs/independent_search/grounding.py`：`_GROUNDING_PROMPT`、`verify_target_grounding`、`TargetInstanceRegistry`。 |
| 证据集合、联合包及接受门 | `qavs/independent_search/decision.py`、`bundles.py`。 |
| 二分类否定门 | `qavs/independent_search/binary.py`：`StrictNoConfig`、`evaluate_negative_gate`。 |
| 条件标签损失接口与精确输入复用 | `qavs/evidence_gap/pdf_runtime.py`：`wrapper_option_label_losses`、`CachedOptionLabelLosses`。 |
| 模型加载、视觉编码及生成 | `qavs/models/modeling_dispatch.py` 及对应模型适配器。 |

计算复用仅复用相同模型与完整相同输入下的确定性结果。不同选项仍分别评分；缓存不会把不同视图算作同一份证据，也不改变逻辑预算。在支持的配置中，没有数据依赖的生成与验证可以并行执行。

## H. 单次搜索伪代码

```text
输入 I, q, O，以及冻结模型和固定配置
生成并保存 y_global；计算全图信息充分性及逐选项验证
若启用且通过全图快捷门：返回验证首选（存在性问题仅 Yes）
若严格全图门通过且不是二分类 No：返回全图答案
构建无答案问题计划、SAM 候选、排序队列及状态
必要时 SCAN 补充 SGAP 候选
while 仍有预算且存在新观察：
    生成当前答案；逐选项验证；独立定位并关联实例
    更新当前分支证据，或建立符合角色/实例要求的联合包
    按固定聚合/共识配置计算接受条件
    若候选是 No：同时检查否定覆盖与反驳门
    若所有适用接受条件通过：冻结答案并立即返回
    按证据缺口执行观察动作，或 RECOVER 后重新筛选证据
返回预先保存的 y_global
```
