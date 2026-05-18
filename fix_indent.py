with open('medical_chatbot/agents/patient_agent.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
for line in lines:
    # Fix the under-indented session print line
    if '[session] context=' in line and line.startswith('    print'):
        line = '        print(f"[session] context updated")\n'
    fixed.append(line)

with open('medical_chatbot/agents/patient_agent.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed)

print('Done')