# -*- coding: utf-8 -*-
"""
R0-T001 Legacy Claim Provenance & Reproducibility Audit
========================================================
对旧版 Uniswap V3 AI Hunter 的 README 核心结论做证据溯源审计：
1. 扫描 README 与旧脚本中的关键数值；
2. 输出 Claim Matrix（结论证据矩阵）；
3. 标记数值来源分类（DIRECT_COMPUTE / OPTIMIZER_OUTPUT / HARD_CODED /
   HEURISTIC_ADJUSTMENT / MANUAL_SUMMARY / UNVERIFIED）；
4. 标记训练 / 验证区间重叠风险（STRICT_OOS / OVERLAP / IN_SAMPLE / UNKNOWN）；
5. 输出 JSON 与 Markdown 两种结果；
6. 对模型元数据做受控读取（不依赖 deap / demeter / optuna，缺依赖时降级 UNVERIFIED）；
7. 不依赖链上写权限，不修改任何旧脚本 / 原始数据。

仅允许修改：results/r0_t001/ 下输出文件。
"""

import json
import os
import re
import sys
import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 分类标签（协议 CURRENT_TASK.md §8）
# ---------------------------------------------------------------------------
DIRECT_COMPUTE = "DIRECT_COMPUTE"          # 代码直接从数据计算
OPTIMIZER_OUTPUT = "OPTIMIZER_OUTPUT"      # 优化器输出
HARD_CODED = "HARD_CODED"                  # 硬编码常数或固定结果
HEURISTIC_ADJUSTMENT = "HEURISTIC_ADJUSTMENT"  # 经验系数 / 人工修正
MANUAL_SUMMARY = "MANUAL_SUMMARY"          # README 人工汇总，代码找不到完整证据
UNVERIFIED = "UNVERIFIED"                  # 当前无法验证

STRICT_OOS = "STRICT_OOS"   # 严格样本外
OVERLAP = "OVERLAP"         # 搜索 / 训练和验证存在重叠
IN_SAMPLE = "IN_SAMPLE"     # 完全样本内
UNKNOWN = "UNKNOWN"         # 代码 / 数据不足以判断


# ---------------------------------------------------------------------------
# 核心结论定义：README 数值 + 需要建立的证据链（CURRENT_TASK.md §5）
# ---------------------------------------------------------------------------
def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return f.read()
        except Exception:
            return ""


def _grep(text, pattern):
    """返回 pattern 在文本中所有出现的行号列表（1-based）。"""
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line):
            hits.append(i)
    return hits


def _grep_all_files(files, pattern):
    """返回 {相对路径: [行号...]}。"""
    out = {}
    for rel in files:
        p = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(p):
            hits = _grep(_read_text(p), pattern)
            if hits:
                out[rel] = hits
    return out


def _analyze_model_metadata(model_path=None):
    """
    受控读取 models_15m.pkl 元数据。
    优先用 pickle.load（需 xgboost）；缺失则降级用 pickletools 静态解析；
    都不可用时标记 UNVERIFIED。绝不执行链上写 / 交易。

    参数 model_path：可选，模型文件绝对/相对路径。供测试注入不存在的路径。
    """
    if model_path is None:
        pkl_rel = "v3_experimental_15m_tag/models_15m.pkl"
        pkl_path = os.path.join(REPO_ROOT, pkl_rel)
    else:
        pkl_path = model_path
    result = {"path": pkl_path, "exists": os.path.isfile(pkl_path)}

    if not result["exists"]:
        result["status"] = "UNVERIFIED"
        result["reason"] = "model file not found"
        return result

    # --- 方式 A：完整 pickle.load（需要 xgboost；deap 缺失时 ga 会失败） ---
    try:
        import pickle
        with open(pkl_path, "rb") as f:
            m = pickle.load(f)
        keys = list(m.keys()) if isinstance(m, dict) else []
        result["top_level_keys"] = keys
        if "xgb" in keys:
            x = m["xgb"]
            result["model_class"] = type(x).__name__
            result["model_module"] = type(x).__module__
            try:
                params = x.get_params() if hasattr(x, "get_params") else {}
                keep = {k: v for k, v in params.items()
                        if not callable(v) and not isinstance(v, (dict, list))}
                result["xgb_params"] = keep
            except Exception as e:
                result["xgb_params_error"] = str(e)
        if "features" in keys:
            feats = m["features"]
            result["feature_count"] = len(feats)
            result["features"] = list(feats)
        if "ga" in keys:
            ga = m["ga"]
            try:
                result["ga_values"] = list(ga)
                result["ga_type"] = type(ga).__name__
            except Exception as e:
                result["ga_values_error"] = str(e)
        result["status"] = "LOADED" if "top_level_keys" in result else "PARTIAL"
        return result
    except ImportError as ie:
        missing = str(ie)
        # 缺少依赖，降级为 pickletools 静态解析
        result["load_error"] = f"ImportError: {missing}"
        result["status"] = "DEGRADED"
    except Exception as e:
        result["load_error"] = f"{type(e).__name__}: {e}"
        result["status"] = "DEGRADED"

    # --- 方式 B：pickletools 静态解析（不依赖 deap / xgboost） ---
    try:
        import io
        import pickletools
        with open(pkl_path, "rb") as f:
            data = f.read()
        strings = []
        floats = []
        for op, arg, pos in pickletools.genops(io.BytesIO(data)):
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                strings.append(arg)
            elif op.name in ("BINFLOAT", "FLOAT"):
                floats.append(arg)

        result["static_strings"] = strings
        result["static_float_count"] = len(floats)
        result["static_floats"] = floats

        # 提取模型类名：寻找 'xgboost.sklearn' 之后紧跟的类名（STACK_GLOBAL 语义）
        for i, s in enumerate(strings):
            if s == "xgboost.sklearn" and i + 1 < len(strings):
                result["model_class"] = strings[i + 1]
                result["model_module"] = "xgboost.sklearn"
                break
        # 若上面没命中，找 'XGBClassifier' 字样
        if "model_class" not in result and "XGBClassifier" in strings:
            result["model_class"] = "XGBClassifier"
            result["model_module"] = "xgboost.sklearn"

        # 提取顶层键（dict 的第一个键是 'xgb'，之后 'ga'、'features'）
        top_keys = []
        for s in strings:
            if s in ("xgb", "ga", "features"):
                top_keys.append(s)
        result["top_level_keys"] = top_keys

        # 提取 features（'features' 之后直到非特征名的短字符串）
        feats = []
        in_feat = False
        stop = {"deap", "creator", "Individual", "fitness", "FitnessMax",
                "deap.base", "Fitness", "weights", "wvalues",
                "numpy._core.multiarray", "scalar", "numpy", "dtype", "f8", "<"}
        for s in strings:
            if s == "features":
                in_feat = True
                continue
            if in_feat:
                if s in stop:
                    break
                feats.append(s)
        result["features"] = feats
        result["feature_count"] = len(feats)

        # 提取 GA 参数：deap Individual 段之后出现的浮点值
        # （顶层键顺序为 xgb, ga, features；GA 段位于 features 之前、deap 段之后）
        ga_vals = []
        in_deap_section = False
        for op, arg, pos in pickletools.genops(io.BytesIO(data)):
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE") and arg == "deap.creator":
                in_deap_section = True
                continue
            if in_deap_section and op.name in ("BINFLOAT", "FLOAT"):
                v = arg
                if v == v and abs(v) < 1e6:  # 过滤 NaN 与异常值
                    ga_vals.append(v)
        # 排除 fitness 权重值（deap 通常为 1.0 / -1.0 的适应度权重），保留实际参数
        ga_params = [v for v in ga_vals if abs(v) not in (0.0, 1.0)]
        if ga_params:
            result["ga_values"] = ga_params
        result["ga_fitness_like"] = [v for v in ga_vals if abs(v) in (0.0, 1.0)]
        result["status"] = "DEGRADED_STATIC"
        return result
    except Exception as e:
        result["static_error"] = f"{type(e).__name__}: {e}"
        result["status"] = "UNVERIFIED"
        result["reason"] = "model could not be parsed"
        return result


# ---------------------------------------------------------------------------
# Claim Matrix 构建
# ---------------------------------------------------------------------------
def build_claim_matrix():
    files_py = [
        "lp_smart_agent.py",
        "dual_engine_optimizer.py",
        "wide_range_study.py",
        "demeter_asymmetric_backtest.py",
        "v3_raw_reality_check.py",
        "v3_hunter_monte_carlo.py",
    ]
    readme_text = _read_text(os.path.join(REPO_ROOT, "README.md"))
    scripts = {f: _read_text(os.path.join(REPO_ROOT, f)) for f in files_py}
    all_py_text = "\n".join(scripts.values())

    claims = []

    def add(claim_id, claim, readme_value, source_desc, data_desc,
            classification, oos, credibility, rerun, evidence):
        claims.append({
            "claim_id": claim_id,
            "claim": claim,
            "readme_value": readme_value,
            "code_source": source_desc,
            "data_source": data_desc,
            "classification": classification,
            "oos_status": oos,
            "credibility": credibility,
            "recommend_rerun": rerun,
            "evidence": evidence,
        })

    # --- 1. RANGE_PCT = ±8.13% ---
    hits_813 = {**{f: _grep(t, r"0\.0813|8\.13") for f, t in scripts.items() if _grep(t, r"0\.0813|8\.13")},
                "README.md": _grep(readme_text, r"8\.13")}
    add(
        "R0-T001-C1",
        "RANGE_PCT（做市区间）来源",
        "±8.13%（约 25 倍资本效率）",
        "wide_golden_params.pkl = {range: 0.081264, risk_thresh: 0.568, m_bull: 52, m_bear: 50}，由 wide_range_study.py 的 Optuna 搜索（range∈[0.08,0.12]）得到；lp_smart_agent.py 硬编码 RANGE_PCT=0.0813",
        "本地 UNIV3_DATA minute.csv（598 天）；但脚本 DATA_DIR='uniswap_data/UNIV3_DATA' 在本仓库不存在",
        OPTIMIZER_OUTPUT,  # 来自优化器输出；在 lp_smart_agent 中表现为 HARD_CODED
        IN_SAMPLE,
        "部分可信（有优化器产物，但搜索=验证同源）",
        True,
        {"wide_golden_params.pkl": "range=0.08126424856562808",
         "wide_range_study.py": "objective(): range∈[0.08,0.12]",
         "lp_smart_agent.py": "RANGE_PCT = 0.0813",
         "hits": hits_813},
    )

    # --- 2. XGB_RISK_THRESHOLD = 0.57 ---
    hits_057 = {f: _grep(t, r"0\.57|0\.568") for f, t in scripts.items() if _grep(t, r"0\.57|0\.568")}
    add(
        "R0-T001-C2",
        "XGB_RISK_THRESHOLD（风险报警阈值）来源",
        "0.57（过滤 90% 以上随机噪音）",
        "wide_golden_params.pkl risk_thresh=0.568 来自 wide_range_study.py Optuna 搜索（risk_thresh∈[0.40,0.70]）；lp_smart_agent.py 硬编码 0.57。注意：demeter_asymmetric_backtest.py 用 0.45，v3_hunter_monte_carlo.py 用 0.55，脚本间阈值不一致",
        "同上",
        OPTIMIZER_OUTPUT,
        IN_SAMPLE,
        "部分可信",
        True,
        {"wide_golden_params.pkl": "risk_thresh=0.5679913691225329",
         "lp_smart_agent.py": "XGB_RISK_THRESHOLD = 0.57",
         "demeter_asymmetric_backtest.py": "risk_signal xgb_prob>0.45",
         "v3_hunter_monte_carlo.py": "is_risk xgb_prob>0.55",
         "hits": hits_057},
    )

    # --- 3. 4 天再平衡冷却期 ---
    hits_4d = {f: _grep(t, r"days=4|timedelta\(days=4\)") for f, t in scripts.items() if _grep(t, r"days=4")}
    add(
        "R0-T001-C3",
        "4 天再平衡冷却期来源",
        "4 天（用于锁定手续费复利）",
        "dual_engine_optimizer.py / wide_range_study.py / demeter_asymmetric_backtest.py 中硬编码 timedelta(days=4)；lp_smart_agent.py 硬编码 REBALANCE_DELAY_DAYS=4",
        "无（代码常量）",
        HARD_CODED,
        IN_SAMPLE,
        "可信（硬编码一致）",
        False,
        {"dual_engine_optimizer.py": "timedelta(days=4)",
         "wide_range_study.py": "timedelta(days=4)",
         "demeter_asymmetric_backtest.py": "timedelta(days=4)",
         "lp_smart_agent.py": "REBALANCE_DELAY_DAYS = 4",
         "hits": hits_4d},
    )

    # --- 4. 最终净值 $29,270 ---
    add(
        "R0-T001-C4",
        "最终净值 $29,270 来源",
        "$29,270（AI 猎手最终净值）",
        "仅 README.md 出现；所有旧脚本 / pkl / 结果文件均无 29,270 值。dual_engine_optimizer.py 有 ROI 输出但未保存该净值，且基准 hardcode 20863",
        "无法溯源",
        MANUAL_SUMMARY,
        UNKNOWN,
        "不可信 / 无法复现",
        True,
        {"README.md": "最终净值 $29,270",
         "absent": "代码中无 29270 常量或输出"},
    )

    # --- 5. 总 ROI +40.3% ---
    add(
        "R0-T001-C5",
        "总 ROI +40.3% 来源",
        "+40.3%（总 ROI）",
        "仅 README.md。dual_engine_optimizer.py: roi=(final_nav/20863-1)*100，基准 20863 为硬编码，非代码计算值；无 40.3 常量",
        "无法溯源",
        MANUAL_SUMMARY,
        UNKNOWN,
        "不可信 / 无法复现",
        True,
        {"README.md": "总 ROI +40.3%",
         "dual_engine_optimizer.py": "roi = (final_nav/20863 - 1)*100  # 硬编码基准",
         "absent": "代码中无 40.3 常量或输出"},
    )

    # --- 6. 相对 Alpha +45.3% ---
    add(
        "R0-T001-C6",
        "相对 Alpha +45.3% 来源",
        "+45.3%（相对 Alpha）",
        "仅 README.md。代码中无 45.3 常量或输出",
        "无法溯源",
        MANUAL_SUMMARY,
        UNKNOWN,
        "不可信 / 无法复现",
        True,
        {"README.md": "相对 Alpha +45.3%",
         "absent": "代码中无 45.3"},
    )

    # --- 7. Monte Carlo 胜率 91.7% ---
    add(
        "R0-T001-C7",
        "Monte Carlo 胜率 91.7% 来源",
        "91.7%（蒙特卡罗胜率）",
        "仅 README.md。v3_hunter_monte_carlo.py 只跑 10 次随机（range(10)），SUCCESS=(res_df>0).mean()*100；10 次随机不可能得 91.7%（9/10=90%, 10/10=100%），与代码不匹配",
        "本地 raw.csv（部分）",
        MANUAL_SUMMARY,
        OVERLAP,
        "不可信 / 与代码不匹配",
        True,
        {"v3_hunter_monte_carlo.py": "for i in range(10) / SUCCESS mean",
         "README.md": "蒙特卡罗胜率 91.7%",
         "note": "91.7% 无法由 10 次运行产生（非 10 的整数倍）"},
    )

    # --- 8. +40.44% / +47.65% / +32.88% / +24.15% ---
    add(
        "R0-T001-C8a",
        "+40.44% 来源",
        "+40.44%（任务引用数值）",
        "当前仓库代码 / README / git 历史均无 40.44",
        "无",
        UNVERIFIED,
        UNKNOWN,
        "不可信",
        True,
        {"absent": "代码与 README 中无 40.44"},
    )
    add(
        "R0-T001-C8b",
        "+47.65% 来源",
        "+47.65%（Alpha，注释声称 bear market 验证）",
        "仅 lp_smart_agent.py 第 18 行注释 'Alpha +47.65% verified in bear market'，无任何计算代码支撑",
        "无",
        MANUAL_SUMMARY,
        UNKNOWN,
        "不可信 / 无计算支撑",
        True,
        {"lp_smart_agent.py": "# Optimized Wide-Range Parameters (Alpha +47.65% verified in bear market)"},
    )
    add(
        "R0-T001-C8c",
        "+32.88% 来源",
        "+32.88%（Raw Reality Check 输入）",
        "v3_raw_reality_check.py: final_roi_raw = 32.88 * 0.85（硬编码）+ 0.85 'Reality Penalty' 经验系数",
        "raw.csv（但核心结果硬编码）",
        HARD_CODED,
        IN_SAMPLE,
        "不可信 / 估算冒充回测",
        True,
        {"v3_raw_reality_check.py": "final_roi_raw = 32.88 * 0.85  # The 'Reality Penalty'"},
    )
    add(
        "R0-T001-C8d",
        "+24.15% 来源",
        "+24.15%（任务引用数值）",
        "当前仓库代码 / README / git 历史均无 24.15",
        "无",
        UNVERIFIED,
        UNKNOWN,
        "不可信",
        True,
        {"absent": "代码与 README 中无 24.15"},
    )

    # --- 9. 原子级 / Raw Log 回测是否真正逐笔 ---
    add(
        "R0-T001-C9",
        "原子级 / Raw Log 回测是否真正逐笔计算",
        "README 声称解析几十 GB 链上原始 Swap Log 捕捉插针",
        "v3_raw_reality_check.py 读取 raw.csv 并循环 swap，但：p_entry/p_low/p_high/L 初始为 0 且从未更新；state 永远 POOL（state=='ETH' 分支不可达）；PnL 计算主体为 pass；注释明示 'Local Feature Proxy'、'for the sake of this massive run'；最终 ROI=32.88*0.85 硬编码",
        "本地 raw.csv（598 天，~15 GB）",
        HEURISTIC_ADJUSTMENT,
        IN_SAMPLE,
        "不可信 / 非真正逐笔回测",
        True,
        {"v3_raw_reality_check.py": "p_entry,p_low,p_high,L = 0,0,0,0 (never set)",
         "v3_raw_reality_check.py": "pass  # OUT OF BOUNDS - STOP EARNING FEES",
         "v3_raw_reality_check.py": "final_roi_raw = 32.88 * 0.85"},
    )

    # --- 10. 15 秒延迟 / 5 bps 滑点 / Reality Penalty 是否真实进入计算 ---
    add(
        "R0-T001-C10",
        "现实约束（15 秒延迟 / 5bps 滑点 / Reality Penalty）是否真实进入计算",
        "README 声称显式引入 5s 采样 + 10s 上链确认延迟惩罚",
        "5bps：dual_engine_optimizer.py / wide_range_study.py latency_bias=0.0005 作为固定滑点（经验值，非实测）；15 秒延迟：仅 v3_raw_reality_check.py 注释提及，无实现；Reality Penalty：0.85 硬编码乘数",
        "无真实延迟 / 滑点数据",
        HEURISTIC_ADJUSTMENT,
        IN_SAMPLE,
        "不可信 / 均为经验假设",
        True,
        {"dual_engine_optimizer.py": "latency_bias = 0.0005",
         "wide_range_study.py": "latency_bias = 0.0005",
         "v3_raw_reality_check.py": "0.85 # The 'Reality Penalty'",
         "v3_raw_reality_check.py": "comment: Trade happens 15 seconds LATER"},
    )

    return claims


# ---------------------------------------------------------------------------
# 训练 / 验证泄漏分析（CURRENT_TASK.md §7）
# ---------------------------------------------------------------------------
def build_leakage_matrix():
    leaks = []
    leaks.append({
        "script": "wide_range_study.py",
        "search_range": "range∈[0.08,0.12], risk_thresh∈[0.40,0.70], m_bull∈[50,65], m_bear∈[35,50]",
        "validation_window": "同一 full_minute_df 全量数据 + 2025-08-24 起 Peak Start Stress Test",
        "overlap": "是（搜索与验证用同一份全量数据）",
        "future_data": "否（未显式用未来，但无时间分割）",
        "strict_oos": IN_SAMPLE,
    })
    leaks.append({
        "script": "dual_engine_optimizer.py",
        "search_range": "range∈[0.02,0.05], risk_thresh∈[0.40,0.65]",
        "validation_window": "search_df=full_minute_df.iloc[-260000:]（最近 ~6 个月），final=full_minute_df 全量",
        "overlap": "是（最终验证包含搜索用的最后 6 个月）",
        "future_data": "否",
        "strict_oos": OVERLAP,
    })
    leaks.append({
        "script": "demeter_asymmetric_backtest.py",
        "search_range": "无搜索（固定 ±4% 区间 + 固定阈值 0.45）",
        "validation_window": "全量 365 天",
        "overlap": "不适用（无参数搜索，但无独立验证集）",
        "future_data": "否",
        "strict_oos": IN_SAMPLE,
    })
    leaks.append({
        "script": "v3_hunter_monte_carlo.py",
        "search_range": "无参数搜索（固定 0.55 / EMA / RSI 阈值）",
        "validation_window": "10 次随机 25-35 天窗口",
        "overlap": "技术指标由历史价格滚动/重采样计算，脚本用 merge_asof(direction='backward') 并入既有信号，未发现明确 look-ahead；但 models_15m.pkl 训练来源（训练脚本/标签/窗口/切分）缺失，无法判定随机测试窗口是否属于严格样本外",
        "future_data": "UNKNOWN（技术指标管线未发现明确 look-ahead，但模型训练窗口未知）",
        "strict_oos": UNKNOWN,
    })
    leaks.append({
        "script": "v3_raw_reality_check.py",
        "search_range": "无",
        "validation_window": "声称全量 raw，实际硬编码结果",
        "overlap": "不适用（结果非计算产生）",
        "future_data": "否",
        "strict_oos": IN_SAMPLE,
    })
    return leaks


# ---------------------------------------------------------------------------
# 数据路径映射（CURRENT_TASK.md §4）
# ---------------------------------------------------------------------------
def build_data_mapping():
    old_ref = "uniswap_data/UNIV3_DATA"
    actual_candidates = [
        "D:\\gitee\\uniswap-data\\UNIV3_DATA",
        "C:\\Users\\peter\\Documents\\V3_Strategy\\UNIV3_DATA",
    ]
    return {
        "old_script_data_dir": old_ref,
        "old_script_data_dir_exists_in_repo": os.path.isdir(os.path.join(REPO_ROOT, old_ref)),
        "old_script_refs": {
            "demeter_asymmetric_backtest.py": "DATA_DIR = 'uniswap_data/UNIV3_DATA'",
            "dual_engine_optimizer.py": "DATA_DIR = 'uniswap_data/UNIV3_DATA'",
            "wide_range_study.py": "DATA_DIR = 'uniswap_data/UNIV3_DATA'",
            "v3_raw_reality_check.py": "DATA_DIR = 'uniswap_data/UNIV3_DATA'",
            "v3_hunter_monte_carlo.py": "DATA_DIR = 'uniswap_data/UNIV3_DATA'",
        },
        "actual_local_roots": [c for c in actual_candidates if os.path.isdir(c)],
        "gitignore_excludes_uniswap_data": "uniswap_data/",
        "conclusion": "旧脚本引用的相对路径在本仓库 checkout 内不存在（.gitignore 忽略 uniswap_data/），因此从仓库根直接运行时无法读取数据；本机实际数据位于 D:\\gitee\\uniswap-data\\UNIV3_DATA（Harness 配置根）",
    }


# ---------------------------------------------------------------------------
# 模型可复现性结论
# ---------------------------------------------------------------------------
def build_model_reproducibility(model_meta):
    train_scripts = ["train", "training", "fit", "create_model"]
    has_train_script = False
    for f in os.listdir(REPO_ROOT):
        if f.endswith(".py") and any(k in f.lower() for k in train_scripts):
            has_train_script = True
    return {
        "model_meta": model_meta,
        "training_script_in_repo": has_train_script,
        "reproducible_from_repo": False,
        "missing_to_reproduce": [
            "模型训练脚本（仓库中不存在能生成 models_15m.pkl 的脚本）",
            "标签定义（LVR 风险标签如何从数据生成）",
            "训练窗口 / 数据版本",
            "随机种子 / 早停 / 验证划分",
        ],
        "conclusion": "models_15m.pkl 无法从当前仓库代码 + 本地数据完整重建；缺少训练脚本与标签定义。",
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    out_dir = os.path.join(REPO_ROOT, "results", "r0_t001")
    os.makedirs(out_dir, exist_ok=True)

    claims = build_claim_matrix()
    leaks = build_leakage_matrix()
    data_map = build_data_mapping()
    model_meta = _analyze_model_metadata()
    model_repro = build_model_reproducibility(model_meta)

    report = {
        "report_id": "R0-T001-legacy-claim-audit",
        "generated_at": datetime.datetime.now().isoformat(),
        "remote_head_consumed": "04ac458c4154ceb5780980a1b2c2eb45c0f6b54b",
        "claim_matrix": claims,
        "leakage_matrix": leaks,
        "data_mapping": data_map,
        "model_reproducibility": model_repro,
        "conclusions": {
            "readme_403_trustworthy": False,
            "mc_917_reproducible": False,
            "raw_atomic_real": False,
            "model_retrainable": False,
            "next_rerun_recommendation": "最值得重跑：wide_range_study.py 的 wide-range（±8.13%）策略在严格 OOS 分割下做全量逐笔回测（该参数有优化器产物与脚本支撑，是唯一有完整代码路径的核心结论）；其次为 v3_hunter_monte_carlo.py 增加独立样本外信号重建。",
        },
    }

    # JSON
    json_path = os.path.join(out_dir, "legacy_claim_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Markdown
    md = render_markdown(claims, leaks, data_map, model_repro, report["conclusions"])
    md_path = os.path.join(out_dir, "legacy_claim_audit.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("claims:", len(claims))
    return 0


def render_markdown(claims, leaks, data_map, model_repro, conclusions):
    lines = []
    lines.append("# R0-T001 Legacy Claim Audit（旧系统结论证据审计）\n")
    lines.append(f"- 生成时间：{datetime.datetime.now().isoformat()}")
    lines.append(f"- 消费 remote_head：04ac458c4154ceb5780980a1b2c2eb45c0f6b54b\n")

    lines.append("## Claim Matrix（结论证据矩阵）\n")
    lines.append("| Claim / 结论 | README 数值 | 代码来源 | 数据来源 | 分类 | OOS 状态 | 当前可信度 | 是否建议重跑 |")
    lines.append("|---|---:|---|---|---|---|---|---|")
    for c in claims:
        lines.append(f"| {c['claim']} | {c['readme_value']} | {c['code_source'][:120]} | {c['data_source'][:60]} | {c['classification']} | {c['oos_status']} | {c['credibility']} | {'是' if c['recommend_rerun'] else '否'} |")
    lines.append("")

    lines.append("## 训练 / 验证泄漏分析\n")
    lines.append("| 脚本 | 搜索区间 | 验证窗口 | 重叠 | 未来数据 | 结论 |")
    lines.append("|---|---|---|---|---|---|")
    for l in leaks:
        lines.append(f"| {l['script']} | {l['search_range']} | {l['validation_window']} | {l['overlap']} | {l['future_data']} | {l['strict_oos']} |")
    lines.append("")

    lines.append("## 数据路径映射\n")
    lines.append(f"- 旧脚本引用：`{data_map['old_script_data_dir']}`")
    lines.append(f"- 本仓库内是否存在：{data_map['old_script_data_dir_exists_in_repo']}")
    lines.append(f"- 本机实际数据根：{', '.join(data_map['actual_local_roots'])}")
    lines.append(f"- 结论：{data_map['conclusion']}\n")

    lines.append("## 模型可复现性\n")
    mm = model_repro["model_meta"]
    lines.append(f"- 模型文件存在：{mm.get('exists')}")
    lines.append(f"- 读取状态：{mm.get('status')}")
    if mm.get("top_level_keys"):
        lines.append(f"- 顶层键：{mm.get('top_level_keys')}")
    if mm.get("model_class"):
        lines.append(f"- 模型类：{mm['model_module']}.{mm['model_class']}")
    if mm.get("feature_count"):
        lines.append(f"- 特征数：{mm['feature_count']}")
        lines.append(f"- 特征：{', '.join(mm.get('features', [])[:6])} ...")
    if mm.get("ga_values"):
        lines.append(f"- GA 参数：{mm['ga_values']}")
    lines.append(f"- 仓库内训练脚本：{model_repro['training_script_in_repo']}")
    lines.append(f"- 可完整重建：{model_repro['reproducible_from_repo']}")
    if model_repro.get("missing_to_reproduce"):
        lines.append("- 缺失项：")
        for m in model_repro["missing_to_reproduce"]:
            lines.append(f"  - {m}")
    lines.append("")

    lines.append("## 总体结论\n")
    for k, v in conclusions.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
