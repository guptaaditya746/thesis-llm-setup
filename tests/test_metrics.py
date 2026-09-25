from llm_setup.metrics import parse_metrics


def test_metrics_parse_and_absent_values_are_safe():
    parsed = parse_metrics('''
vllm:num_requests_running 2
vllm:num_requests_waiting 3
vllm:kv_cache_usage_perc 82
vllm:request_queue_time_seconds_bucket{le="1"} 9
vllm:request_queue_time_seconds_bucket{le="5"} 10
vllm:request_queue_time_seconds_bucket{le="+Inf"} 10
''')
    assert parsed["running"] == 2 and parsed["waiting"] == 3
    assert parsed["kvCache"] == .82 and parsed["queueP95Seconds"] == 5
    assert "latencyP95Seconds" in parsed and parsed["latencyP95Seconds"] is None


def test_preemptions_are_reported_when_present():
    parsed = parse_metrics('vllm:num_preemptions_total{engine="0",model_name="heavy-model"} 7.0\n')
    assert parsed["preemptions"] == 7
    assert "preemptions" not in parse_metrics("vllm:num_requests_running 1\n")
