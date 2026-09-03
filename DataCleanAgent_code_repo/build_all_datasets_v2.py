import csv
import os
import glob
import random

random.seed(42)

# ===== 路径配置 =====
DATA_DIR = r"C:\Users\27878\WorkBuddy\20260408170411\datasets\med_79w\Chinese-medical-dialogue-data-master\Data_数据"
ANNOTATION_FILE = r"C:\Users\27878\WorkBuddy\20260408170411\outputs\annotation\annotation_100_v4.csv"
OUTPUT_DIR = r"C:\Users\27878\WorkBuddy\20260408170411\outputs\experiment"

# ===== 步骤1：合并所有科室数据 =====
print("=" * 60)
print("[1/4] 合并所有科室数据...")

all_rows = []  # [(department, question, answer)]
dept_counts = {}

for dept_dir in sorted(os.listdir(DATA_DIR)):
    dept_path = os.path.join(DATA_DIR, dept_dir)
    if not os.path.isdir(dept_path):
        continue
    dept_name = dept_dir.replace('_', '')
    count = 0
    for csv_file in sorted(glob.glob(os.path.join(dept_path, '*.csv'))):
        try:
            with open(csv_file, 'r', encoding='gbk', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = (row.get('title', '') + ' ' + row.get('ask', '')).strip()
                    answer = row.get('answer', '').strip()
                    all_rows.append((dept_name, question, answer))
                    count += 1
        except Exception as e:
            print("  [WARN] 跳过 %s: %s" % (csv_file, e))
    dept_counts[dept_name] = count
    print("  %s: %d 条" % (dept_name, count))

print("  合计: %d 条" % len(all_rows))

# 保存79W总数据
total_file = os.path.join(OUTPUT_DIR, 'data_total_79w.csv')
with open(total_file, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['row_index', 'department', 'question', 'answer'])
    for i, (dept, q, a) in enumerate(all_rows):
        writer.writerow([i, dept, q, a])
print("  -> 已保存: %s" % total_file)

# ===== 步骤2：按比例抽样10W条 =====
print("\n" + "=" * 60)
print("[2/4] 生成10W条实验数据（按比例抽样）...")

sample_10w = random.sample(all_rows, 100000)
print("  抽样: 100000 条")

exp_10w_file = os.path.join(OUTPUT_DIR, 'data_exp_10w.csv')
with open(exp_10w_file, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['row_index', 'department', 'question', 'answer'])
    for i, (dept, q, a) in enumerate(sample_10w):
        writer.writerow([i, dept, q, a])
print("  -> 已保存: %s" % exp_10w_file)

# ===== 步骤3：按比例抽样1000条 =====
print("\n" + "=" * 60)
print("[3/4] 生成1000条测试数据（按比例抽样）...")

sample_1k = random.sample(all_rows, 1000)
print("  抽样: 1000 条")

test_1k_file = os.path.join(OUTPUT_DIR, 'data_test_1k.csv')
with open(test_1k_file, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['row_index', 'department', 'question', 'answer'])
    for i, (dept, q, a) in enumerate(sample_1k):
        writer.writerow([i, dept, q, a])
print("  -> 已保存: %s" % test_1k_file)

# ===== 步骤4：替换前100条（保持department不变） =====
print("\n" + "=" * 60)
print("[4/4] 将专家标注替换到前100行（保持department不变）...")

# 读取专家标注
annotations = []
with open(ANNOTATION_FILE, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row.get('患者提问', '').strip()
        a = row.get('医生回答', '').strip()
        val = row.get('【专家填写】是否有噪声(0=无/1=有)', '').strip()
        label = int(val) if val in ('0', '1') else ''
        annotations.append({'question': q, 'answer': a, 'expert_label': label})

print("  已读取 %d 条专家标注" % len(annotations))

def replace_first_n_rows(filepath, annotations):
    """替换前N条，只替换question/answer/department，保留原始department"""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    for i, ann in enumerate(annotations):
        if i < len(rows):
            # row_index不变，department不变，question和answer替换为专家标注
            rows[i][2] = ann['question']
            rows[i][3] = ann['answer']

    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # 验证前3条
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        print("  Header: %s" % header)
        for i in range(3):
            row = next(reader)
            print("  Row %d: dept=%s, q=%s..." % (i+1, row[1], row[2][:30]))

replace_first_n_rows(exp_10w_file, annotations)
replace_first_n_rows(test_1k_file, annotations)

print("\n" + "=" * 60)
print("完成！")
print("  1. 总数据: %s (%d 条)" % (total_file, len(all_rows)))
print("  2. 10W实验: %s (前100条question/answer已替换为专家标注，department保持原样)" % exp_10w_file)
print("  3. 1000测试: %s (前100条question/answer已替换为专家标注，department保持原样)" % test_1k_file)
print("=" * 60)
