# 代码守则

> 适用于：任何提交代码、创建分支、合并 PR 的场景。
> 
> 核心原则：**小步快跑、频繁提交、禁止直接推 main。**

---

## 🌿 分支与提交规则

1. **一个提交只做一件逻辑变更**。如果 commit message 需要"和"字，说明提交太大，必须拆分。
2. **禁止直接推 main**。所有改动走 feature branch。
3. **大重构必须拆成可独立合并的小片**（垂直切片）。一个分支只做一件事，做完就合并，禁止长期 hanging branch。
4. **有进展就要 commit**。没提交的代码等于没做，push 即是备份也是沟通。禁止在本地堆积大量未提交变更。

---

## 代码质量红线

- 新代码必须带测试（详见 [STANDARDS_TESTING_BUGS.md](STANDARDS_TESTING_BUGS.md)）
- 修改代码必须同步更新文档（详见 [STANDARDS_DOCUMENTATION.md](STANDARDS_DOCUMENTATION.md)）
- 发现测试失败或 bug 必须立即处理（详见 [STANDARDS_TESTING_BUGS.md](STANDARDS_TESTING_BUGS.md)）

---

*返回总纲：[CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)*
