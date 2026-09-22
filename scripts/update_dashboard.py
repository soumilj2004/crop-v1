with open(r'C:\CropGuardAI\cropguard_ai\scripts\training_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update phase ID and name
content = content.replace(
    '{"id": "0_bootstrap", "name": "Data Bootstrap", "icon": "📦", "required": True,',
    '{"id": "0_leakproof", "name": "Leakproof Bootstrap", "icon": "shield", "required": True,'
)
content = content.replace(
    '"desc": "Auto-download base wheat data, create splits, merge LWDCD2020"},',
    '"desc": "Zero-leakage pipeline: dedup + label clean + verified splits"},'
)

# Update button handlers
content = content.replace(
    'if st.button("📦 Merge Data (Phase 0)",',
    'if st.button("Leakproof Bootstrap",'
)
content = content.replace(
    'st.session_state.current_phase = "0_data_prep"',
    'st.session_state.current_phase = "0_leakproof"'
)
content = content.replace(
    'threading.Thread(target=run_phase_bg, args=("0_data_prep", False), daemon=True).start()',
    'threading.Thread(target=run_phase_bg, args=("0_leakproof", False), daemon=True).start()'
)

with open(r'C:\CropGuardAI\cropguard_ai\scripts\training_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Dashboard updated')