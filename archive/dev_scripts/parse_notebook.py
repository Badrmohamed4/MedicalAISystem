import json

with open('skinDisease case/FINAL_Skin_Diseases.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'class_names' in source or 'classes' in source or 'dataset' in source:
            print("--- SOURCE ---")
            print(source)
            print("--- OUTPUT ---")
            for output in cell.get('outputs', []):
                if 'text' in output:
                    print("".join(output['text']))
                elif 'data' in output and 'text/plain' in output['data']:
                    print("".join(output['data']['text/plain']))
