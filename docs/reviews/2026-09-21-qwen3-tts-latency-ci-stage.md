# 外审:给 Qwen3-TTS CI 臂加一个流式首帧延迟阶段

- 时间:2026-09-21 16:45 PT
- 模型档位:GPT-6 Pro(临时聊天,思考 7m30s)
- 材料:`tests/test_model/tts_ci_config.py`、`.github/workflows/test-tts-ci.yaml`、`test_tts_ci.py` 的骨架、
  `benchmarks/metrics/performance.py` 的指标定义、run doc 第二十七到三十一轮
- 我的原判断见文末

## 它说了什么

总判断:**"先当粗粒度的流式回归护栏发出去;亚毫秒的判定另做一套配对基准。这是两件事,想用一套阈值同时做只会得到要么抖、要么误导的结果。"**

### 一、它指出的问题,我认可并采纳

1. **只 gate 完成率不是延迟护栏**。我自己的混合 prefill 实验就是反例:rps 1 不变、rps 20 从 40 ms 变 2.7 秒,
   请求最终都在超时内完成,完成率照样 100%。→ 第一天起就把"20 rps 的 `audio_ttfp_median_s`"列为第一个要校准的 timing gate,
   在它校准好之前把这个阶段称作"带性能报告的正确性 gate",不叫回归护栏。
2. **要证明开环没有退化成闭环**。`audio_ttfp_*` 从发送时刻起算,不是从计划到达时刻;客户端的 semaphore 或连接池等待会从指标里消失。
   `--concurrency 16` 不能被继承成客户端准入上限,否则服务变慢时 offered load 跟着降,回归被藏起来。
   → 记录每条请求的计划到达、实际发送、首包、完成;单独报告 dispatch lateness;用预生成的带 seed 的 Poisson 到达序列,
   对照序列而不是要求实际速率恰好等于 20;撞上安全上限要判 run 无效,不能静默限流。
3. **首个解码块不等于可用音频**。`first_audio_payload_bytes_mean`、`audio_chunks_mean`、underrun 分位、c50/c100/c200 这些字段要一起看,
   否则一个"先发一个极小的首包然后卡住"的实现能刷分。→ 第一天起要求解码音频非空、流正常结束、多块请求的 continuity 覆盖率显式检查,报首包时长而不只是字节。
4. **单 worker**。现有 fixture 明确起两个 worker 且断言两个都被用到;20 rps 分到两个 worker 就是每个 10 rps,不是我在优化的工作点。
   → 单独的 fixture / 显式拓扑参数,保留 router,断言恰好一个健康 worker 且它服务了被测请求;不改共享 fixture 的默认值。
5. **worst-of-N 不是低误报的保证**。独立同分布下第六次超过前五次最大值的概率就是 1/6。
   → 先定 job 级统计量(三个预定 seed 的 run-p50 取中位数),按这个统计量校准;阈值 = 校准 job 的最大值 + 显式的加性余量 M,
   M 用同 build 的留出 run 和要抓的回归量级定;用 A/A 看误报、用注入的首块延迟看漏报,两者都要过。
6. **`run_flaky_pytest.sh` 的重试不能替性能结果做选择**。保留所有 attempt,用预定的聚合规则。
7. **第二十七轮那张表加起来 8.73 ms,标题写的 9.09 ms**,差的 0.36 ms 正好是要判的优化量级。→ 表要标明各 span 定义和残差,不当成可加的关键路径。
8. **CPU 计时不是 GPU 计时**。`graph.replay()` 外面套 host timer 量的是发射开销;要用 CUDA event 夹住图、事后收集,
   用有界的 event 池 + 延迟收集,不在每步同步。并且"随便什么批次的平均重放时间"不是可靠的微优化指标,候选可能改变批次构成;
   要按(batch、模式、序列长度、dtype、图配置)分层比较,以"predictor 每帧时间"这类语义单元为主指标。
9. **标签触发留下覆盖空洞**。调度器/router 的公共改动不带 Qwen 标签也会影响 Qwen3-TTS。→ 默认分支定期跑,公共 serving 路径改动要触发。
10. 60 条样本的 p95 只有约 3 个观测在上面,不要一开始就 gate 低负载 p95;冻结一个有代表性的 60 条子集并版本化,不要默认取语料前 60 条。

### 二、它反对但我坚持的

无。它没有反对我的方向,只是把"回归护栏"和"小优化裁判"拆成了两件事,这一点我原来就模糊,接受。

### 三、它同意的

复用仓库里现成的 benchmark 与指标;两个工作点(1 与 20 rps)都保留;未校准前 report-only;单 worker。

## 我的原判断(发审前写下,原样保留)

见 `my-judgment.md` 的内容:复用现成 benchmark、单 worker、两个工作点、未校准只打印、完成率第一天 gate;
最不确定的是要不要加机制级指标,以及单 worker 还是沿用双 worker fixture。
外审对这两点都给了明确答案:机制级指标要做但要用 CUDA event、按批次分层、单独成一套配对基准;单 worker。
# 我自己的判断(发外审之前写下)

## 要做的东西
给 Qwen3-TTS 的 CI 臂加一个"流式首帧延迟"阶段,作为后续优化的常设量尺。

## 设计
1. 复用仓库里现成的 `benchmarks/eval/benchmark_tts_seedtts`:它已有开环到达(`request_rate`,指数间隔)、流式首包时间
   `audio_ttfp_{median,p95,p99}_s`、播放连续性(c50/c100/c200 通过率、underrun p95)。不引入新的客户端。
2. 新 pytest 阶段 `tts-stage-latency`,只在 `TTS_CI_MODEL` 是 qwen3-tts / qwen3-tts-custom-voice 时由 workflow 触发;
   单 worker 挂在 router 后面(`num_workers=1`),因为要量的是单服务的首帧,两 worker 会把负载对半分、还混进 router 的均衡噪声。
3. 两个工作点:1 rps(60 条)与 20 rps(全量 1088 条),流式 PCM,先预热。
4. 判据按 #2094 定下的规矩:未校准的 preset 只打印不 gate;等 CI 主机上按 worst-of-N 校准后再开 gate。
   唯一从第一天就 gate 的是确定性不变量:完成率 100%。
5. 阈值字段放进 `TtsCiThresholdPreset`(新增 `latency` 子结构),沿用 `apply_slack`。

## 我认为的风险
- 端到端首帧在共享主机上 A/A 漂移 0.1-1.5 ms,而剩余候选单项只有 0.2-0.8 ms:这个阶段**抓不住小改进**,只能抓大回归
  (比如 CUDA graph runner 被静默禁用、混合 prefill 那种 40 ms→2.7 s)。它的定位是"回归护栏 + 趋势记录",不是小优化的裁判。
- CI 时长 +8-10 分钟(多起一次服务)。
- 20 rps 对 CI runner 是否过载未知(runner 是 H100,我的数字来自 H100,应当可行)。

## 我最不确定的
- 是否应该再加一个"机制级"指标(decode 步里 talker/predictor 图的重放时间),它抖动小、能裁判小优化,
  但需要服务端暴露计时,改动面更大。
- 单 worker 还是沿用两 worker 的现有 fixture(省一次启动)。
