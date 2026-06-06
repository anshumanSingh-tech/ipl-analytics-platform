# fix_callbacks.py — run from project root
with open("dashboard/callbacks.py", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split("\n")
print("Lines with empty string column references:")
for i, line in enumerate(lines, 1):
    if '[""]' in line or "['']" in line:
        print(f"  Line {i}: {line.strip()}")

print()

# Order matters: Longest/most specific strings must be replaced FIRST
replacements = [
    ('rename(columns={"": "player"})', 'rename(columns={"batsman": "player"})'),
    ('rename(columns={"":',            'rename(columns={"batsman":'),
    ('[""] not in index',  '["batsman"] not in index'),
    ('"", "total_runs"',   '"batsman", "total_runs"'),
    ('"", "balls_bowled"', '"bowler", "balls_bowled"'),
    ('batting[""]',        'batting["batsman"]'),
    ('bowling[""]',        'bowling["bowler"]'),
    ('plot_df[""]',        'plot_df["batsman"]'),
    ('[""].apply',   '["batsman"].apply'),
    ('[""].unique',  '["batsman"].unique'),
    ('[""].iloc',    '["batsman"].iloc'),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"Fixed: {old!r}  →  {new!r}")

with open("dashboard/callbacks.py", "w", encoding="utf-8") as f:
    f.write(content)

print("\nDone. Verifying no empty strings remain...")
remaining = [f"Line {i}: {l.strip()}"
             for i, l in enumerate(content.split("\n"), 1)
             if '[""]' in l or "['']" in l]
if remaining:
    print("Still found:")
    for r in remaining:
        print(f"  {r}")
else:
    print("All clear — no empty string column references remain.")