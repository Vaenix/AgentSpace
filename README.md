# AgentSpace

Playground repository for the dsh-superagent experiment.

## 长视频中重复声画事件的证据定位

同一种声音在视频里多次出现。当问题从“第一次之后做了什么”换成“第二次之后做了什么”，视频问答模型能否找到对应时段，并正确描述随后发生的动作？

| 研究材料 | 内容 |
|---|---|
| [实验方案](av-occurrence-study/实验方案.md) | 研究假设、对照组、抽样与评分规则 |
| [执行记录](av-occurrence-study/执行记录.md) | 2026 年 11 月 19 日与 24 日的实测结果、样本限制与未跑指标 |
| [独立复测](av-occurrence-study/run/replication_1124/README.md) | 3 部新视频、6 道题、18 次推理，提供冻结输入、逐题回复与统计 |
| [整片检索对照](av-occurrence-study/data/retrieval_1124/visual/视觉检索.md) | 在同一部完整视频上比较纯视觉检索与声画联合检索 |
| [盲核与评分](av-occurrence-study/eval/README.md) | 从人工独立复核到冻结参考答案和评分的流程 |

当前批次暂无人工核准的正式题对，准确率和定位命中率均为**未跑**。行为复测衡量改变选项或片段代号后，模型是否改变引用位置。

原始输出和统计表可直接查阅。需要播放或重新推理时，请按数据方条款获取媒体与模型权重。EPIC 官方标注的下载地址、目标文件名和 SHA-256 见[来源清单](av-occurrence-study/epic/provenance.json)。将标注存入 `av-occurrence-study/epic/annotations/`，其余素材准备见 [EPIC 数据说明](av-occurrence-study/epic/数据说明.md)和[本地推理说明](av-occurrence-study/run/README.md)。[盲核页面](av-occurrence-study/review.html)需在本地浏览器打开，页面内提供原片链接。
