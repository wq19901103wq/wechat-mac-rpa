#!/usr/bin/env python3
"""同步 knowledge_source.md → aliases.json / facts.json / corrections.json / wiki 文件

用法：
    python3 scripts/sync_knowledge.py

读取 data/memory/knowledge_source.md，解析后自动更新：
- data/memory/overrides/aliases.json
- data/memory/overrides/facts.json
- data/memory/overrides/corrections.json
- data/memory/wiki/users/*.md（把 facts 合并进去）
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


def parse_knowledge_source(md_path: Path) -> dict:
    """解析 markdown，返回 {people: [...], groups: [...]}"""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    people: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None  # 'people' or 'groups'

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(">") or stripped.startswith("---"):
            continue

        # 顶级标题判断当前段落类型（必须在一级标题判断之前处理）
        if stripped in ("# 人物", "# 人物知识"):
            section = "people"
            current = None
            continue
        if stripped in ("# 群聊", "# 群聊知识"):
            section = "groups"
            current = None
            continue

        # 跳过其他一级标题（如文档标题）
        if re.match(r"^#[^#]", stripped):
            continue

        # 二级标题 = 具体人物/群聊名
        m = re.match(r"^##\s+(.+)$", stripped)
        if m:
            name = m.group(1).strip()
            current = {"name": name, "aliases": [], "facts": []}
            if section == "people":
                people.append(current)
            elif section == "groups":
                groups.append(current)
            continue

        # 别名行
        if stripped.startswith("别名：") or stripped.startswith("别名:"):
            if current is not None:
                alias_text = stripped.split("：", 1)[1].strip() if "：" in stripped else stripped.split(":", 1)[1].strip()
                if alias_text and alias_text != "无":
                    current["aliases"] = [a.strip() for a in alias_text.split(",") if a.strip()]
            continue

        # bullet points = 知识点
        m = re.match(r"^[-*]\s+(.+)$", stripped)
        if m and current is not None:
            current["facts"].append(m.group(1).strip())

    return {"people": people, "groups": groups}


def extract_relation_value(fact_text: str) -> dict:
    """从自然语言 fact 提取 relation + value + note
    支持格式：
      - 表哥是王海，昵称小海哥，大舅家的表哥
      - 喜欢现场看球，支持泰州队
      - 做AI视频生成相关工作
    """
    fact_text = fact_text.strip().rstrip("。")

    # 尝试提取 "X是Y" 或 "X：Y" 模式
    m = re.match(r"^(.+?)[是:：]\s*(.+)$", fact_text)
    if m:
        relation = m.group(1).strip()
        value = m.group(2).strip()
        # 把逗号后的内容作为 note
        if "，" in value:
            parts = value.split("，", 1)
            value = parts[0].strip()
            note = parts[1].strip()
        else:
            note = ""
        return {"relation": relation, "value": value, "note": note}

    # 无法提取关系时，用整句作为 value
    return {"relation": "备注", "value": fact_text, "note": ""}


def update_aliases(people: list, aliases_path: Path):
    """更新 aliases.json"""
    data: dict[str, Any] = {"users": {}}
    if aliases_path.exists():
        try:
            data = json.loads(aliases_path.read_text(encoding="utf-8"))
        except Exception as e:
            _logger.warning("load aliases failed: %s", e)

    for p in people:
        name = p["name"]
        if name not in data.get("users", {}):
            data.setdefault("users", {})[name] = {"aliases": [], "notes": ""}
        # 合并别名（去重）
        existing = set(data["users"][name].get("aliases", []))
        for alias in p["aliases"]:
            existing.add(alias)
        data["users"][name]["aliases"] = list(existing)
        # 用第一条 fact 作为 notes（如果没有的话）
        if p["facts"] and not data["users"][name].get("notes"):
            data["users"][name]["notes"] = p["facts"][0][:100]

    aliases_path.parent.mkdir(parents=True, exist_ok=True)
    aliases_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  [aliases] 更新了 {len(people)} 个人物")


def update_facts(people: list, facts_path: Path):
    """更新 facts.json"""
    data: dict[str, Any] = {"users": {}}
    if facts_path.exists():
        try:
            data = json.loads(facts_path.read_text(encoding="utf-8"))
        except Exception as e:
            _logger.warning("load facts failed: %s", e)

    for p in people:
        name = p["name"]
        if not p["facts"]:
            continue
        facts = []
        for fact_text in p["facts"]:
            extracted = extract_relation_value(fact_text)
            facts.append(extracted)
        data.setdefault("users", {})[name] = {"facts": facts}

    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  [facts] 更新了 {sum(1 for p in people if p['facts'])} 个人物")


def update_corrections(groups: list, corrections_path: Path):
    """更新 corrections.json"""
    data: dict[str, Any] = {"groups": {}}
    if corrections_path.exists():
        try:
            data = json.loads(corrections_path.read_text(encoding="utf-8"))
        except Exception as e:
            _logger.warning("load corrections failed: %s", e)

    for g in groups:
        name = g["name"]
        if not g["facts"]:
            continue
        data.setdefault("groups", {})[name] = {"corrections": g["facts"]}

    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    corrections_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  [corrections] 更新了 {len(groups)} 个群聊")


def update_wiki_files(people: list, wiki_dir: Path):
    """把 facts 合并到对应的 wiki .md 文件中"""
    users_dir = wiki_dir / "users"
    users_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    for p in people:
        name = p["name"]
        path = users_dir / f"{name}.md"

        # 读取现有 wiki（如果有）
        if path.exists():
            wiki = path.read_text(encoding="utf-8")
        else:
            wiki = f"# {name}\n\n## 基本信息\n（暂无）\n\n## 偏好 & 兴趣\n（暂无）\n\n## 近期动态\n（暂无）\n\n## 说过的话（短期）\n（暂无）\n\n## 交互风格\n（暂无）\n"

        # 如果 wiki 中已经有 "## 补充信息（人工标注）" 段落，替换它
        # 否则在文件末尾添加
        facts_md = "## 补充信息（人工标注）\n"
        for fact_text in p["facts"]:
            extracted = extract_relation_value(fact_text)
            line = f"- {extracted['relation']}：{extracted['value']}"
            if extracted["note"]:
                line += f"\n  （{extracted['note']}）"
            facts_md += line + "\n"

        if "## 补充信息（人工标注）" in wiki:
            # 替换现有段落
            wiki = re.sub(
                r"## 补充信息（人工标注）.*?\n(?=## |\Z)",
                facts_md + "\n",
                wiki,
                flags=re.DOTALL,
            )
        else:
            # 在文件末尾添加
            wiki = wiki.rstrip() + "\n\n" + facts_md

        path.write_text(wiki, encoding="utf-8")
        updated += 1

    print(f"  [wiki] 更新了 {updated} 个用户 wiki 文件")


def sync() -> bool:
    """同步 knowledge_source.md → 所有配置和 wiki 文件。
    返回是否实际执行了同步（文件存在且有内容）。
    """
    base = Path("data/memory")
    md_path = base / "knowledge_source.md"

    if not md_path.exists():
        return False

    parsed = parse_knowledge_source(md_path)
    if not parsed["people"] and not parsed["groups"]:
        return False

    overrides_dir = base / "overrides"
    wiki_dir = base / "wiki"

    update_aliases(parsed["people"], overrides_dir / "aliases.json")
    update_facts(parsed["people"], overrides_dir / "facts.json")
    update_corrections(parsed["groups"], overrides_dir / "corrections.json")
    update_wiki_files(parsed["people"], wiki_dir)
    return True


def main():
    base = Path("data/memory")
    md_path = base / "knowledge_source.md"

    if not md_path.exists():
        print(f"错误：{md_path} 不存在")
        return

    print(f"读取: {md_path}")
    parsed = parse_knowledge_source(md_path)
    print(f"解析到: {len(parsed['people'])} 个人物, {len(parsed['groups'])} 个群聊")
    print()

    overrides_dir = base / "overrides"
    wiki_dir = base / "wiki"

    update_aliases(parsed["people"], overrides_dir / "aliases.json")
    update_facts(parsed["people"], overrides_dir / "facts.json")
    update_corrections(parsed["groups"], overrides_dir / "corrections.json")
    update_wiki_files(parsed["people"], wiki_dir)

    print()
    print("同步完成！")


if __name__ == "__main__":
    main()
