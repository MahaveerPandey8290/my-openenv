from .models import OrderTestAction, SubmitTriageAction, PatientObservation, TriageState

__all__ = ["OrderTestAction", "SubmitTriageAction", "PatientObservation", "TriageState"]


def get_client():
    """Lazy import to avoid triggering openenv import chain at server startup."""
    from .client import ClinicalTriageEnvClient
    return ClinicalTriageEnvClient
