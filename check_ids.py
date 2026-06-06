import re

def get_ids():
    try:
        with open('dashboard/layout.py', encoding='utf-8') as f:
            layout_ids = set(re.findall(r'id=[\"\'](.*?)[\"\']', f.read()))

        with open('dashboard/callbacks.py', encoding='utf-8') as f:
            cb_content = f.read()
            output_ids = set(re.findall(r'Output\([\"\'](.*?)[\"\']', cb_content))
            input_ids  = set(re.findall(r'Input\([\"\'](.*?)[\"\']', cb_content))

        print(f"Found {len(layout_ids)} IDs in layout.")
        print(f"Found {len(output_ids)} Output IDs in callbacks.\n")

        missing = output_ids - layout_ids
        if missing:
            print('❌ ERROR: The following Output IDs are not in your layout:')
            for i in sorted(missing): print(f'  - {i}')
        else:
            print('✅ SUCCESS: All callback IDs exist in layout.')

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")

if __name__ == "__main__":
    get_ids()