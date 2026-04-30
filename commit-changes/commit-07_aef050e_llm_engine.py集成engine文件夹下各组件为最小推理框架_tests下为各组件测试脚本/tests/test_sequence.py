import pytest

from engine.structs import Sequence, SequenceStatus


def test_sequence_status_full_transition_chain():
    seq = Sequence(request_id="req-state", input_ids=[1, 2, 3])
    assert seq.status == SequenceStatus.WAITING

    seq.transition_to(SequenceStatus.RUNNING_PREFILL)
    seq.transition_to(SequenceStatus.RUNNING_DECODE)
    seq.transition_to(SequenceStatus.FINISHED)

    assert seq.status == SequenceStatus.FINISHED


def test_sequence_invalid_transition_raises():
    seq = Sequence(request_id="req-invalid", input_ids=[1])
    with pytest.raises(ValueError):
        seq.transition_to(SequenceStatus.FINISHED)
