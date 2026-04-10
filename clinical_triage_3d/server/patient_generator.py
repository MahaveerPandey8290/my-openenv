# Copy patient_generator and test_bank from v3 for the 3D environment
from clinical_triage_env.server.patient_generator import generate_patient
from clinical_triage_env.server.test_bank import get_test_result

__all__ = ["generate_patient", "get_test_result"]
