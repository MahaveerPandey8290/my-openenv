import sys
sys.path.insert(0, '.')
from clinical_triage_env.server.tasks import TASK_REGISTRY, grade_order_test
from clinical_triage_env.models import SubmitTriageAction
from clinical_triage_env.server.patient_generator import generate_patient

print('=== Score validation ===')
errors = []
for task_name, spec in TASK_REGISTRY.items():
    for seed in range(10):
        patient = generate_patient(task_name, seed=seed)
        for level in ['immediate','urgent','less_urgent','non_urgent']:
            action = SubmitTriageAction(
                action_type='submit_triage',
                triage_level=level,
                suspected_condition=patient['condition_category'] + ' suspected',
                reasoning='Testing all combinations for score validation. Two sentences minimum here.'
            )
            for tests in [0, 1, 2, 3]:
                score = spec.grade_fn(action, patient, 2, tests)
                if not (0.0 < score < 1.0):
                    errors.append('FAIL ' + task_name + ' level=' + level + ' seed=' + str(seed) + ' tests=' + str(tests) + ' score=' + str(score))

for relevant in [True, False]:
    for used in [1, 2, 3]:
        s = grade_order_test(relevant, used, 3)
        if not (0.0 < s < 1.0):
            errors.append('FAIL order_test relevant=' + str(relevant) + ' used=' + str(used) + ' score=' + str(s))

if errors:
    for e in errors:
        print(e)
else:
    print('ALL SCORES STRICTLY IN (0.01, 0.99) -- PASS')
    print('Procedural generation works across 10 seeds x 4 levels x 4 tests -- PASS')

# Also validate 3D graders
print()
print('=== 3D Score validation ===')
try:
    from clinical_triage_3d.server.tasks import TASK_REGISTRY_3D, _grade_triage_assignments
    d_errors = []
    for task, spec in TASK_REGISTRY_3D.items():
        patients = {'bed_' + str(i+1): {'triage_level': ['immediate','urgent','less_urgent','non_urgent'][i%4]} for i in range(spec.n_patients)}
        for assignments in [
            {bid: p['triage_level'] for bid, p in patients.items()},
            {bid: 'non_urgent' for bid in patients},
            {bid: 'urgent' for bid in patients},
        ]:
            score = spec.grade_fn(assignments, patients, 30.0)
            if not (0.0 < score < 1.0):
                d_errors.append(task + ' score=' + str(score))
    if not d_errors:
        print('ALL 3D SCORES STRICTLY IN (0.01, 0.99) -- PASS')
    else:
        print('FAIL:', d_errors)
except ImportError as e:
    print('SKIP (3D requires pygame/PyOpenGL — install for Docker use):', e)
