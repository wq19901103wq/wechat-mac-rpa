#!/bin/bash
# 微信 OCR V4 测试脚本

cd "$(dirname "$0")/.."

echo "=================================="
echo "🧪 微信 Vision OCR V4 测试套件"
echo "=================================="
echo ""

# 检查 fixtures
if [ ! -d "tests/fixtures" ] || [ -z "$(ls -A tests/fixtures/*.json 2>/dev/null)" ]; then
    echo "⚠️  未找到测试用例"
    echo "请先运行: python3 tests/prepare_fixtures.py"
    exit 1
fi

# 运行测试
echo "开始测试..."
echo ""

python3 -c "
import sys
sys.path.insert(0, '.')
from tests.test_ocr_v4 import WeChatOCRTestCase, WeChatVisionOCRBotV4, test_error_cases
from pathlib import Path
import json

bot = WeChatVisionOCRBotV4()
fixture_dir = Path('tests/fixtures')

# 1. 运行正式测试用例
test_files = list(fixture_dir.glob('*.json'))
print(f'📁 正式测试用例: {len(test_files)}个')
print('='*60)

results = []
for json_file in sorted(test_files):
    test_name = json_file.stem
    test_case = WeChatOCRTestCase(fixture_dir, test_name)
    success = test_case.run(bot)
    results.append((test_name, success, len(test_case.errors)))

# 汇总正式测试
print('\\n' + '='*60)
print('📊 正式测试结果')
print('='*60)

passed = sum(1 for _, s, _ in results if s)
total = len(results)

for name, success, error_count in results:
    status = '✅' if success else '❌'
    msg = '通过' if success else f'{error_count} 个错误'
    print(f'{status} {name}: {msg}')

print(f'\\n正式测试: {passed}/{total} 通过 ({passed/total*100:.1f}%)')

# 2. 运行错误案例回归测试
error_results = test_error_cases(bot)

# 最终汇总
print('\\n' + '='*60)
print('📊 最终汇总')
print('='*60)
all_passed = passed
all_total = total

if error_results:
    error_fixed = sum(1 for _, s, _ in error_results if s)
    error_total = len(error_results)
    all_passed += error_fixed
    all_total += error_total
    print(f'正式测试: {passed}/{total} 通过')
    print(f'错误案例: {error_fixed}/{error_total} 已修复')
    print(f'总计: {all_passed}/{all_total} ({all_passed/all_total*100:.1f}%)')
else:
    print(f'总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)')

if all_passed == all_total:
    print('\\n🎉 所有测试通过！')
else:
    print(f'\\n⚠️  {all_total - all_passed} 个失败')
    exit(1)
"
