with open(r'C:\CropGuardAI\cropguard_ai\scripts\training_orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '"id": "0_bootstrap",',
    '"id": "0_leakproof",'
)
content = content.replace(
    '"name": "Data Bootstrap",',
    '"name": "Leakproof Bootstrap",'
)
content = content.replace(
    '"script": "scripts/bootstrap_data.py",',
    '"script": "scripts/leakproof_bootstrap.py",'
)

with open(r'C:\CropGuardAI\cropguard_ai\scripts\training_orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Orchestrator updated')