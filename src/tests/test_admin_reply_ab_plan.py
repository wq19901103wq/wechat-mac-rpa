import scripts.admin as admin


class _EmptyConnection:
    def execute(self, *_args, **_kwargs):
        return self

    def fetchall(self):
        return []

    def close(self):
        pass


class _EmptyDb:
    def _get_conn(self):
        return _EmptyConnection()


def test_reply_ab_plan_states_single_variable_and_quality_gates():
    body = admin._reply_ab_plan_html()

    for expected in (
        "待确认",
        "当前 C0",
        "同聊天、2026 年 4 月前",
        "同 chat 目前只加 1.5 分",
        "先汇报差异",
        "完整上下文",
        "群聊 70% / 私聊 30%",
        "多人连续发言与对象识别",
        "A 更好 / B 更好 / 相当",
        "新增事实幻觉",
        "新增真实行动承诺",
        "不能单独决定 A/B 胜负",
        "私聊不得明显退步",
    ):
        assert expected in body
    assert "[{" not in body
    assert "is_badcase" not in body


def test_experiments_page_keeps_plan_when_no_experiment(monkeypatch):
    monkeypatch.setattr(admin, "get_db", lambda: _EmptyDb())

    response = admin.experiments_list()
    body = response.body.decode("utf-8")

    assert "首轮回复质量 A/B 方案" in body
    assert "暂无已执行实验" in body
    assert '<a href="/experiments" class="active">' in body
    assert body.count('class="card"') > 0


def test_adversarial_plan_is_mobile_readable_and_non_executing():
    body = admin._adversarial_plan_html()

    for expected in (
        "方案已确认",
        "D-Jtrain",
        "D-Btrain",
        "D-dev",
        "D-validation",
        "包含鲁西西",
        "复杂大群必须单独抽样",
        "多个子话题并行",
        "第一阶段首筛集已封存",
        "首筛封存总数",
        "复杂大群",
        "不同聊天来源",
        "第一阶段 · 120 条首筛",
        "第二阶段 · 240 条验证与流程诊断",
        "正确共同富裕群至少 20 个",
        "柚子群至少 20 个",
        "指定群例外",
        "验证与流程诊断集：覆盖已构建",
        "诊断集总数",
        "群聊<br>目标 160 · 已满足",
        "复杂大群<br>目标 100 · 已超过 2",
        "正确共同富裕群<br>目标 20 · 已满足",
        "柚子群<br>目标 20 · 已满足",
        "240</strong><span>独立事件",
        "仅柚子群最多放宽至 2026-07-31 23:59:59",
        "实际封存数据只用到 2026-07-08",
        "封存 20 个不同事件",
        "公开聚合核对：20/20 · 缺口 0",
        "Stage 2 私有 manifest SHA256",
        "扫描已知曝光文件：5498 个",
        "确定性的保守词法规则",
        "多人并行",
        "现实承诺",
        "为什么首筛集的指定群仍是 0",
        "事件簇隔离：通过",
        "模型调用：0",
        "双侧 sign test p&lt;0.05",
        "净胜率至少提升 10 个百分点",
        "至少高 5 个百分点",
        "新增事实幻觉 = 0",
        "新增现实行动承诺 = 0",
        "P95 延迟不超过 C0 10%",
        "每道题一次全新的独立模型 API 调用",
        "风格 Judge 不开放工具",
        "任何一项无法对齐",
        "模型调用：0",
        "C0 修改：无",
        "Bot 启动：无",
        "严格生产可比为 0/240",
        "上下文仅有 speaker/content",
    ):
        assert expected in body
    assert "@media(max-width:700px)" in body
    assert "缺 72" not in body
    assert "缺 35" not in body
    assert "当前硬阻塞" not in body
    assert body.count('class="adv-case"') == 0
    assert "正式 hidden：0/100" not in body
    assert "chat_de96f4621f" not in body
    assert "membership_status" not in body
    assert "source_path" not in body
    assert "source_chat_id" not in body
    assert "case_ids" not in body


def test_adversarial_sample_plan_hides_case_content_and_ids():
    body = admin._adversarial_plan_html()

    for private_value in (
        "internal_id",
        "source_path",
        "event_cluster",
        "5e236406db771064",
        "你还加老王的",
        "b172d01381ee686b",
        "堡黄出来干掉他",
        "chat_de96f4621f",
    ):
        assert private_value not in body
    assert "120" in body
    assert "240" in body
    assert "第二阶段验证与流程诊断集" in body
    assert "80" in body
    assert "50" in body
    assert "40" in body
    assert "方案已确认" in body


def test_adversarial_plan_route_uses_admin_shell():
    response = admin.adversarial_plan()
    body = response.body.decode("utf-8")

    assert "Judge–Bot 对抗实验蓝图" in body
    assert '<a href="/experiments" class="active">' in body
    assert '<meta name="viewport"' in body
    assert body.count("<nav>") == 1
    assert body.count("<main>") == 1
    assert body.count('class="card') > 0


def test_reply_production_alignment_reports_blockers_without_private_cases():
    body = admin._reply_production_alignment_html()

    for expected in (
        "当前不能进入正式 A/B",
        "240",
        "7919",
        "speaker / content",
        "0/240",
        "human_reply",
        "Policy → Store → Generator",
        "同聊天加权",
        "严格重建（已确认）",
        "上下文触发的表达/幽默动作",
        "模型调用：0",
        "Bot 启动：否",
        "C0 修改：否",
        "正式 A/B：禁止",
        "production-single-step-v2",
        "单步回放基础 · 已验证",
        "最小假回放 · 仅用于看格式",
        "甲：明天七点集合？",
        "我先和闹钟谈判",
        "仍然存在的生产差异",
        "20/20",
        "全新合格生产样本",
    ):
        assert expected in body
    for private_value in (
        "chat_de96f4621f",
        "5e236406db771064",
        "你还加老王的",
    ):
        assert private_value not in body
    assert "[{" not in body
    assert "is_badcase" not in body
    assert "@media(max-width:700px)" in body


def test_reply_production_alignment_route_uses_admin_shell():
    response = admin.reply_production_alignment()
    body = response.body.decode("utf-8")

    assert "回复评测生产一致性审计" in body
    assert '<a href="/experiments" class="active">' in body
    assert '<meta name="viewport"' in body
    assert body.count("<nav>") == 1
    assert body.count("<main>") == 1
    assert body.count('class="card') > 0


def test_current_c0_route_uses_formal_240_report():
    response = admin.benchmark_current_c0_historical_ab()
    body = response.body.decode("utf-8")

    assert "C0 固定 240 题正式 A/B" in body
    assert "240/240" in body
    assert "167/240" in body
    assert "163/240" in body
    assert "基线 Judge 判错" in body
    assert "候选 Judge 判错" in body
    assert '<a href="/benchmark/reply/current-c0" class="active">' in body
