# Branch Protection 配置说明

由于 branch protection 规则无法通过仓库文件直接配置，需要在 GitHub 网页端手动开启。

## 配置步骤

1. 打开仓库页面：`https://github.com/example-owner/wechat-mac-rpa`
2. 点击 **Settings** → **Branches**
3. 在 **Branch protection rules** 下点击 **Add rule**
4. **Branch name pattern** 填：`main`（或 `master`）
5. 勾选以下选项：

### 必需配置

- [x] **Require a pull request before merging**
  - [x] **Require approvals**：建议设为 1
  - [x] **Dismiss stale PR approvals when new commits are pushed**
  - [x] **Require review from CODEOWNERS**

- [x] **Require status checks to pass before merging**
  - 搜索并勾选：**CI / test**
  - [x] **Require branches to be up to date before merging**

- [x] **Require conversation resolution before merging**

- [x] **Do not allow bypassing the above settings**

### 建议配置

- [x] **Restrict pushes that create files larger than 100MB**（GitHub 默认）
- [x] **Require signed commits**（可选，如果你有 GPG key）
- [x] **Include administrators**（建议开启，确保规则对所有人生效）

6. 点击 **Create** 或 **Save changes**

## 配置效果

配置完成后：

- 所有人（包括管理员）必须提 PR 才能合并到 main
- PR 必须至少 1 个 CODEOWNERS 审批
- PR 必须通过 CI 测试
- PR 必须有最新代码（需要 rebase/merge 最新 main）
- 所有 review 评论必须解决后才能合并
